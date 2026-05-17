import numpy as np
import random
import sys
from itertools import takewhile, groupby
import mesa
import networkx as nx
import matplotlib.pyplot as plt
import shapely
from shapely.geometry import LineString, Polygon
import copy
from scipy.stats import exponnorm
import re
#two types of agents in the model, BirdsAgent and PoultryAgent

###############################################################################################################################
########################################################### birds #############################################################
###############################################################################################################################
class BirdsAgent(mesa.Agent):
    """
    Bird Agent definition and its properties/interaction methods
    """

    def __init__(
        self,
        unique_id,
        model,
        initial_state,
        experience2diseasepriormigration,
        omega,
        daysalreadywithsymptomprior,
        proba_dying,
        beta_flbf,
        beta_ebf,
        beta_bb,
        excess_CO,
        home_range_area,
        m0,
        mlean, 
        f,
        rc_kgperkm,
        rd_perclean,
        generalproptimeresting, 
        remaining_distance2fly,
        proptimenotflyingtoday,
        proptimeexploring1node,
        proptimerestingtoday,
        symptom,
        expected_MDOS,
        li_states,
        li_pos,
        li_remaining_distance2fly,
        tradeoff,
        proba_startmigration,
        shedding_rate,
        dico_stepid_contaminationpathway,
        S_nf_sites,
        S_nf_farms,
        li_farmpassedabove,
        psi
    ):
        super().__init__(unique_id, model)

        self.state = initial_state

        #virus related
        self.proba_dying = proba_dying
        self.experience2diseasepriormigration = experience2diseasepriormigration
        self.omega = omega
        self.daysalreadywithsymptomprior = daysalreadywithsymptomprior
        
        #bird related
        self.beta_flbf = beta_flbf
        self.beta_ebf = beta_ebf
        self.beta_bb = beta_bb
        self.excess_CO = excess_CO
        self.home_range_area = home_range_area
        self.m0=m0
        self.m=m0
        self.mlean = mlean
        self.f=f
        self.rc_kgperkm = rc_kgperkm
        self.rd_perclean = rd_perclean
        self.generalproptimeresting = generalproptimeresting
        self.remaining_distance2fly = remaining_distance2fly
        self.proptimenotflyingtoday = proptimenotflyingtoday
        self.proptimerestingtoday = proptimerestingtoday #is later: agent.proptimenotflyingtoday*agent.generalproptimeresting
        self.proptimeexploring1node = proptimeexploring1node #is: (agent.proptimenotflyingtoday*(1-agent.generalproptimeresting))/len(S_nf)
        self.symptom = symptom
        self.expected_MDOS = expected_MDOS
        self.li_states = li_states
        self.li_pos = li_pos
        self.li_remaining_distance2fly = li_remaining_distance2fly
        self.tradeoff = tradeoff
        self.proba_startmigration = proba_startmigration
        self.shedding_rate = shedding_rate
        self.dico_stepid_contaminationpathway = dico_stepid_contaminationpathway
        self.S_nf_sites = S_nf_sites
        self.S_nf_farms = S_nf_farms
        self.li_farmpassedabove = li_farmpassedabove
        self.psi = psi
    
    def get_node_attractiveness(self, step_id, source_node_id, node_id, debug=False):
        """get the nodes' 'attractiveness' attribute based on:
        1. the given step ID, ice phenology, mean food availabiltiy score (cant be done for all birds at a same time, 
        since it depends on where the bird is as we should account for how long the bird takes to go there)
        2. bird density (which also cant be done onces per step for all bird)"""
        
        # extract if there will be ice when the bird will be arriving at the site
        ice_start = self.model.G.nodes[node_id]["ice_start"]
        ice_end = self.model.G.nodes[node_id]["ice_end"]
        
        # the case of "sourcenodeid != nodeid" will only be used to evalute whether the bird migrate to that node or not, 
        # and thus when the bird have not yet started the migration journey. Thus, in that case we just compute 
        # the number of days to fly there.
        if source_node_id!=node_id:
            days_flying = (self.model.G.get_edge_data(source_node_id, node_id)['min_distance2patch']*10**3)/self.f
        # if same node (will be used to double check the attractiveness of the node at arrival time / while resting is indeed
        # >0. Thus, the number of days to fly there is stored in the remaining_distance2fly (if more than a full day of 
        # flight, then the attractiveness may be 0 due to ice there, but this is fine, as we wont upate its body mass in 
        # that case anyway)
        else:
            days_flying = self.remaining_distance2fly/self.f
          
        #check if there is still some ice by when the bird would arrive there
        step_id_ = step_id+self.model.DOY+days_flying
        hasnoice_byarrivaltime = 1
        if (self.model.checkice) and ((step_id_>=ice_start) or (step_id_<=ice_end)): 
            hasnoice_byarrivaltime = 0 #has ice!

        # Get alive bird agents located in OR moving towards (i.e. even if meant to arrive later!) nodeid (infected or not) 
        li_agents = self.model.grid.get_cell_list_contents([node_id])
        nbr_bird = len([agent for agent in li_agents if isinstance(agent, BirdsAgent) and agent.state != 'DEAD'])
        bird_density = nbr_bird/(self.model.G.nodes[node_id]["area_size"]*10**6)

        food = self.model.G.nodes[node_id]["mean_foodavailability"] * hasnoice_byarrivaltime
        updated_attract = np.exp(-self.model.birddensity_scaling * bird_density) * food
        if debug:
            print('nbr_bird:', nbr_bird,
                  'area_size:', self.model.G.nodes[node_id]["area_size"] * 10**6,
                  'bird_density:', bird_density,
                  'birddensity_scaling: ',birddensity_scaling,
                  'mean_foodavailability:', self.model.G.nodes[node_id]["mean_foodavailability"],
                  'hasnoice_byarrivaltime:', hasnoice_byarrivaltime,'step_id: ',step_id,'DOY: ',self.model.DOY,
                  'days_flying:',days_flying,'food:', food)
        return round(updated_attract,5)
        
    def update_bodymass(self, PTNF, A, D):
        ''' PTNF: proportion of the day the bird is not flying
            A: Attractiveness of the site
            D: distance2fly over the day'''
        self.m = min(self.m + self.rd_perclean/100*self.mlean*PTNF - self.rc_kgperkm*D, self.m0) #cannot exceed initial body mass
        
    def count_sequences_of_x(self, li, x):
        '''count number of time x appeared in sequence'''
        #groupby: "group consecutive elements of the same value in an iterable object, such as a list"
        return sum(1 for key, group in groupby(li) if key == x)
     #small examples
     #li = ['I','S','S','S','I','I','D','D'] #2
     #['I','S','I','S','S','I','I','D','D']) #3
     #count_sequences_of_x(li=li, x)

    def count_consecutive_x_at_end(self, li, x):
        return sum(1 for _ in takewhile(lambda item: item == x, reversed(li)))
    #small examples:
    #li = ['I','S','S','S','I','I','D','D']
    #count_sequences_of_x(li=li, x='S') #1
    #count_sequences_of_x(li=li, x='I') #2
    #count_consecutive_x_at_end(li=li, x='I') #0
    #count_consecutive_x_at_end(li=li, x='D') #2    
    
    def remove_consecutive_duplicates(self, li):
        return [key for key, _ in groupby(li)]
    #small examples
    #print(remove_consecutive_duplicates(['A','A','A','B','B','C','C','A','B'])) #['A', 'B', 'C', 'A', 'B']
    #print(remove_consecutive_duplicates([1,1,2,2,4,1])) #[1, 2, 4, 1]
    
    def mean_consecutive_runs_exceptfirst(self,li):
        groups = list(sum(1 for _ in g) for _, g in groupby(li))
        #print(groups)
        #remove first as this is before migration
        groups = groups[1:]
        return sum(groups) / len(groups) if groups else 0
    #small example (but not it will be applied to positions, not state)
    #print(mean_consecutive_runs_exceptfirst(['I', 'S','I','S', 'S', 'S', 'I', 'I', 'D', 'D']))  #1.8
    #print(mean_consecutive_runs_exceptfirst(['I', 'I','S', 'S', 'S', 'I', 'I', 'D', 'D']))  #2.33
    #print(mean_consecutive_runs_exceptfirst([]))  #0
    
    def interpolate_coordinates(self, y1, x1, y2, x2, prop):
        #assuming the bird is flying along a straight line
        x = x1 + prop * (x2 - x1)
        y = y1 + prop * (y2 - y1)
        return x, y 
    
    def get_condition_snapshot(self):
        """Collect information on followed birds and their respective nodes"""
        # Get all attributes of the agent as a dictionary (except 'model' to avoid computational load)
        #condition_snapshot = vars(self).copy() #copy.deepcopy(vars(self)) #        
        #del condition_snapshot['model']
        condition_snapshot = {}
        #for the list save it as str() otherwise it will get override by the conditions from the end of the step, and wont represent DC_ 
        #anymore
        for key, value in vars(self).items():
            if key == 'model':
                continue
            if isinstance(value, list):
                condition_snapshot[key] = str(value)
            else:
                condition_snapshot[key] = value
            
        # Retrieve current node information
        current_node = self.pos
        condition_snapshot['nodeid'] = current_node
        #current node is None when the bird is dead
        #current node cant be None since we run this funciton befoer the bird check whether it is dead
        # Add all node attributes to condition_snapshot
        condition_snapshot.update(self.model.G.nodes[current_node])

        #add info from the agents, and then delete this key as its computationally to intense
        num_infected_bird_agents = 0
        num_susceptible_bird_agents = 0
        num_exposed_bird_agents = 0
        num_resistant_bird_agents = 0
        num_dead_bird_agents = 0
        num_bird_agents = 0
        for agent in condition_snapshot['agent']:
            # if the agent is a BirdAgent and in the same node
            if isinstance(agent, BirdsAgent) and agent.pos==self.pos:
                num_bird_agents += 1
                # Check if the bird is infected
                if agent.state == 'INFECTED':
                    num_infected_bird_agents += 1
                if agent.state == 'SUSCEPTIBLE':
                    num_susceptible_bird_agents += 1  
                if agent.state == 'EXPOSED':
                    num_exposed_bird_agents += 1
                if agent.state == 'RESISTANT':
                    num_resistant_bird_agents += 1
                if agent.state == 'DEAD':
                    num_dead_bird_agents += 1
        condition_snapshot['num_onnode_bird_agents'] = num_bird_agents
        condition_snapshot['num_onnode_infected_bird_agents'] = num_infected_bird_agents
        condition_snapshot['num_onnode_susceptible_bird_agents'] = num_susceptible_bird_agents
        condition_snapshot['num_onnode_exposed_bird_agents'] = num_exposed_bird_agents
        condition_snapshot['num_onnode_resistant_bird_agents'] = num_resistant_bird_agents
        condition_snapshot['num_onnode_dead_bird_agents'] = num_dead_bird_agents
        del condition_snapshot['agent']
        #get virus in the environment at that time step
        condition_snapshot['virusinenv_onnode'] = self.model.dico_step_nodeid_virusinenv[self.step_id-1][current_node]
        condition_snapshot['birdidINFO_onnode'] = self.model.dico_step_nodeid_birdidINFO[self.step_id-1][current_node]
        return condition_snapshot
    
    
    def Symptomseffect(self, is_reverse):
        '''is_reverse is boolean. 
        is_reverse = True:  we reverse the effect from infection, i.e. when bird pass from infected to resistent/immune
        is_reversed = False: its during one of exposed or infected state, when he start developping symptoms (i.e.,after the incubation period'''
        
        ##### small check
        #compute the nbr of times bird got infected, where we include one more infection if the last state is 'E' meaning that the bird is exposed, 
        #have its incubation period finished, and thus still not  #infectious, but already 'infected'
        adddays = 1 if self.li_states[-1]=='E' else 0 
        nbr_timeinfected = self.count_sequences_of_x(li=self.li_states, x='I') + self.experience2diseasepriormigration + adddays
        if nbr_timeinfected not in [1,2]:
            print(nbr_timeinfected)
            print(self.li_states)
            raise RuntimeError('ERROR: should only be able to get infected twice at max, check your code')
            
        #after incubation period, starts the symptoms
        if is_reverse==False:
            ##### add the symptoms for these birds that have been infected and finished the incubation period. accounting for naive (first infection) or
            #primes (second infection)
            #Note: psi: no need here, as it only valid while birds are in settelemtn sites, not otherwise, so we use at different time and implemented
            #it directly when needed in the code.           
            hra = self.model.home_range_area_I*self.model.G.graph['max_explorativewithoutrest']*10**3 #for infected birds after incubation period
            #now also accounting for if this is the first infection (and thus naiv birds) or not
            if nbr_timeinfected==1:
                self.home_range_area = hra*self.model.home_range_area_I_scalenaive
                self.generalproptimeresting = self.generalproptimeresting*self.model.generalproptimeresting_I*self.model.generalproptimeresting_I_scalenaive
                self.omega = self.model.omega*self.model.omega_scalenaive
                self.proba_dying = self.model.proba_dying*self.model.proba_dying_scalenaive                
            elif nbr_timeinfected==2:
                self.home_range_area = hra
                self.generalproptimeresting = self.generalproptimeresting*self.model.generalproptimeresting_I
                self.omega = self.model.omega
                self.proba_dying = self.model.proba_dying  
            else:
                print(nbr_timeinfected)
                raise RuntimeError('ERROR nbr_timeinfected is not 1 or 2')              

        #after the symptomatic period, stops the symptoms
        #attention the parameter space should be done so that bird cant get new symptoms while the other one have not been finished.
        elif is_reverse==True:  
            self.home_range_area = self.model.G.graph['max_explorativewithoutrest']*10**3
            self.omega = 0
            self.proba_dying = 0
            #generalproptimeresting needs to start from the birds value ,since their is individdual variation.
            #either the birds was naive 
            if nbr_timeinfected==1:
                self.generalproptimeresting = self.generalproptimeresting/self.model.generalproptimeresting_I/self.model.generalproptimeresting_I_scalenaive
            #or the bird was not naive
            elif nbr_timeinfected==2:
                self.generalproptimeresting = self.generalproptimeresting/self.model.generalproptimeresting_I
            else:
                print(nbr_timeinfected)
                raise RuntimeError('ERROR nbr_timeinfected is not 1 or 2')

        else:
            print('ERROR your is_reverse should be either True or False')
            sys.exit()


    def plot_farm_and_path(self, farm_coords, bird_path_coords):
        """
        Plot the farm rectangle and the bird's path.

        Parameters:
            farm_coords (list of tuple): Four (x, y) coordinates defining the rectangle (farm).
            bird_path_coords (tuple of tuple): Two (x, y) coordinates defining the bird's path.
        """
        print('-----------------------------------------')
        farm_x, farm_y = zip(*farm_coords + [farm_coords[0]])  # Close the rectangle loop
        bird_x, bird_y = zip(*bird_path_coords)
        plt.figure(figsize=(3,2))
        plt.plot(bird_x, bird_y, 'r--', linewidth=1)
        plt.scatter(bird_x, bird_y, color="red", s=1)   # add the path points
        plt.plot(farm_x, farm_y, 'b-', label="Farm", linewidth=1)
        plt.scatter(farm_x, farm_y, color="blue", s=1)  # Farm corners
        #plt.legend()
        plt.title('blue: farm rectangle, red: bird path')
        plt.axis("equal")
        plt.show()
        print('CHECK the red is above the blue!')
    
    def checkInfectionBelowMigratoryRoute(self):
        '''check if a bird flew above a farm, and if so save that info for the farm
        could make it more efficient (e.g., could add on each edge an attribute with the min and max prop of distance that a farm
        intersect)'''
        
        ###### extract some information needed
        # edge flying above (i.e. last to nodes that are different)
        li_e1e2 = self.remove_consecutive_duplicates(self.li_pos)[-2:]
        # number of days the bird has been flying for (which includes this step!)
        nbrdaysflew = self.count_consecutive_x_at_end(li=self.li_pos, x=self.li_pos[-1])
        # total distance to fly
        totaldistance2fly = self.model.G.edges[(li_e1e2[0],li_e1e2[1])]['min_distance2patch']*10**3
        # proportion already done. We have "min" here as a new node can be less than a day away, thus would be > 1
        prop_distanceDone = min(1, (nbrdaysflew-1)*self.f / totaldistance2fly)
        # propotion that will be done by the end of the day
        prop_distanceDoneByEndDay = min(1, nbrdaysflew*self.f / totaldistance2fly)
        
        ###### find initial and end coordinates of the birds flying today using linear interpolation between the 2coordinates
        y1=self.model.G.nodes[li_e1e2[0]]['y']
        x1=self.model.G.nodes[li_e1e2[0]]['x'] 
        y2=self.model.G.nodes[li_e1e2[1]]['y']
        x2=self.model.G.nodes[li_e1e2[1]]['x']
        init_x,init_y = self.interpolate_coordinates(y1=y1, x1=x1, y2=y2, x2=x2, prop=prop_distanceDone)
        end_x,end_y = self.interpolate_coordinates(y1=y1, x1=x1, y2=y2, x2=x2, prop=prop_distanceDoneByEndDay)
        bird_path_coordinates = [(init_x,init_y),(end_x,end_y)]

        ###### check if there are any farms between these coordinates, by iterating over all farms 
        #the reason why we do it like this althought its not efficient, its becaue birds can take several days to fly, and they do so 
        #at different speed, so we would otehrwise need to register for each farm at the begining all the links that the farm is below,
        #and at which proportion of the link.
        li_all_farmagent = [agent for agent in self.model.grid.get_cell_list_contents(self.model.G.nodes()) if \
                            isinstance(agent, PoultryAgent)]
        for farmsagent in li_all_farmagent:

            # farm' rectangle coordinates
            Fmin_x = farmsagent.min_x
            Fmax_x = farmsagent.max_x
            Fmin_y = farmsagent.min_y
            Fmax_y = farmsagent.max_y
            farm_coordinates = [(Fmin_x,Fmin_y),  # Bottom-left corner
                                (Fmin_x, Fmax_y), # Top-left corner
                                (Fmax_x, Fmax_y), # Top-right corner
                                (Fmax_x,Fmin_y)]  # Bottom-right corner
            
            # if the birds passed above the farm save it into the farms attributes
            farm_polygon = Polygon(farm_coordinates)
            bird_path_line = LineString(bird_path_coordinates)
            if shapely.intersects(farm_polygon, bird_path_line):
                #add to farm and to its attribute
                self.li_farmpassedabove.append(self.unique_id)
                farmsagent.libetaflbf_of_Birds_passed_above.append(self.beta_flbf/self.model.sanitation)
                if self.model.printdebug:
                    self.plot_farm_and_path(farm_coords=farm_coordinates, bird_path_coords=bird_path_coordinates)
                    print('last five values of self.li_pos:',self.li_pos[-5:],
                          '\nstep:',self.step_id,
                          '\nlast edge flew above: li_e1e2:', li_e1e2, 
                          '\nnumber of days the bird has been flying for (which includes this step!): nbrdaysflew:',nbrdaysflew, 
                          '\ntotaldistance2fly:', totaldistance2fly,
                          '\ndistance bird can fly in a day:',self.f, 
                          '\nprop_distanceDone:',prop_distanceDone,
                          '\nprop_distanceDoneByEndDay:',prop_distanceDoneByEndDay,
                          '\ninit_x,init_y:',init_x,init_y,
                          '\nend_x,end_y:',end_x,end_y)

            
    ######### behavioural rules
    def Diseasedynamic(self):
        """
        move (loose body mass) & see if death & try infect one virus in the patch
        OR 
        forage (win body mass) & see if death & try to infect farms, all virus in the patch, water
        """
        # Track conditions (both its state and the site caracteristics) before any behavioural action
        if self.model.debuging:
            self.decision_conditions = self.get_condition_snapshot()    
        else: 
            self.decision_conditions = {'remaining_distance2fly':self.remaining_distance2fly, 'nodeid':self.pos}
            
        #save position of each agent for debuging
        agents_on_node = self.model.grid.get_cell_list_contents([self.pos])
        dico_birdid_pos = {}
        for agent in agents_on_node:
            dico_birdid_pos[agent.unique_id] = agent.pos
        if self.model.debuging:
            self.decision_conditions['agentsonnode_4diseasedynamic'] = dico_birdid_pos
                       
        ######################################################################################################
        ############# first a bird update its state: susceptible/infected/resistant(immune)/dead #############
        ######################################################################################################
        
        #no possible infections at step 0 since we assume no time has passed between initial conditions and birds activation
        if self.step_id!=0:

            ########## Check if needs to reverse (one side or the other) symptoms since reversing to normal could happen while the bird is 
            #exposed, infected, resistant, or even susceptible, it is easier to do it before going through each state separately.
            #Note that this logic for th adding Symptomseffect works too for the birds that started infected, as they did not started in an 'E' state and
            #had their AIeffecrt already applied, However, it does not work to reverse the Symptomseffect since they did not started with an 'E' state, so
            #we add this condition as well
            if (self.symptom==1) and (self.state != 'DEAD'):
                
                # first compute the number of days since the first 'E' state from the last sequences of 'E'
                #this is because, depending on the parameter space, the bird could be in any state since beeing exposed once the symptoms starts
                s = ''.join(self.li_states)
                li_m = list(re.finditer(r'E+', s))
                nbrdays_sincefirstE = len(s) - li_m[-1].start() if li_m else 0
                #small example make sure code isright
                # s = "S S E I I R E E E".replace(" ", "")
                # li_m = list(re.finditer(r'E+', s))   
                # print(li_m)                          # two match "EE" starting at index 2 and 6
                # print(li_m[-1].start())              # 6 : take the last block of E
                # print(len(s) - li_m[-1].start())     # 9 - 6 = 3  -> days since first 'E'

                #if incubation period has passed, then INITIATE the Symptomseffect from infection
                #initially infected birds got their symptoms activated, and acccounted for naivety/primed then. In all other cases, they should pass through
                #an Exposed state
                if self.model.days_incubation==nbrdays_sincefirstE:
                    self.Symptomseffect(is_reverse=False) 
                
                #if incubation and symptoms period has passed, then REVERSE the Symptomseffect from infection
                if self.model.days_incubation+self.model.days_symptom==nbrdays_sincefirstE:
                    self.Symptomseffect(is_reverse=True)     

                #add reverse Symptomseffect for birds that started with infection (i.e. after the full smyptom-what has been done, but because its 
                #the day after we should add 1)
                if (self.li_states[0]=='I') and (self.model.days_symptom-self.daysalreadywithsymptomprior+1==len(self.li_states)):
                    self.Symptomseffect(is_reverse=True)     

                # for delayed migration it is accounted for in the initial conditions of the model, for infected birds with symptoms (where we also account
                #for naive birds or not). However, for symptomatic birds that did not started infected, but got exposed prior to moving to a site
                #in which case they should also have delayed start migration (which is still a delay since days 0, and thus when it happen twice, i.e., 
                #if the bird is exposed twice while beeing in the startingsite, this is all fine, wont matter)
                #--> for birds that are still in their starting site, that were not infected at the begining of simulation, that got exposed to AI, 
                #and for which the incubation period has finished, we give them a migration delay.
                if self.model.delayedmigrationinfectedLOC!=-1:
                    if (len(set(self.li_pos))==1) and (self.li_states[0]!='I') and ('E' in self.li_states) and\
                       (self.model.days_incubation==nbrdays_sincefirstE):
                        loc = self.model.delayedmigrationinfectedLOC
                        self.psi = exponnorm.rvs(5, loc=loc, scale=1, size=1)[0]
                        #compute the nbr of times bird got infected, where we include one more infection if the last state is 'E' meaning that the bird is
                        #exposed, have its incubation period finished, and thus still not  #infectious, but already 'infected'
                        adddays = 1 if self.li_states[-1]=='E' else 0 
                        nbr_timeinfected = self.count_sequences_of_x(li=self.li_states, x='I') + self.experience2diseasepriormigration + adddays
                        if nbr_timeinfected==1:
                            self.psi = self.psi*self.model.delayedmigrationinfectedLOC_scalenaive
                        self.psi = 300 if self.psi >= 50 else self.psi
                    
            ########## check if the bird dies (proba should be 0 outside of symptoms period, but a bird could have symptoms while beein infected, exposed,
            #or susceptible, depending on the parameter space)
            #Note: any bird can die while flying!
            #Convert an end-of-period death probability into a per-day probability
            daily_proba_dying = 1 - (1 - self.proba_dying) ** (1 / self.model.days_symptom)
            if (self.random.random() < daily_proba_dying) and (self.symptom==1):
                self.state = 'DEAD'                        
            
            ########## Update the bird's state
            if self.state == 'DEAD':
                pass
            
            elif self.state == 'EXPOSED':
                #if still not enough days Exposed, then stay Exposed (do nothing), else  convert the birds to infected
                nbr_days_withexposed = self.count_consecutive_x_at_end(li=self.li_states, x='E')

                ########## if latent period is finished, then become infectious
                if (nbr_days_withexposed >= self.model.days_exposed):
                    self.state = 'INFECTED'
                    #is actually dead instead of infected if the distance remaining to fly is bigger than what its body mass can sustain,
                    #else its just infected
                    if self.remaining_distance2fly>(self.m-self.mlean)/self.rc_kgperkm:
                        self.state = 'DEAD'
                        print('Interesting: a bird died while flying because he starts getting symptoms!')
                                            
            elif self.state == 'INFECTED':

                ########## if still not enough days infected, then stay infected (do nothing), else check if the bird is dead or resistant/immune
                #At the end of infection period convert its state into resistant (since it survived, i..e state is not dead, but it is infected)
                nbr_days_withinfection = self.count_consecutive_x_at_end(li=self.li_states, x='I')
                if (nbr_days_withinfection >= self.model.days_infected):
                    self.state = 'RESISTANT'

            #if you were resistant, then check if still are resistant, else become susceptible unless you already got 
            #infected twice, in which case the bird stay resistant                    
            #Note that if a bird become susceptible it necessarily mean that it only had one infection (no matter if it happened at the start of the
            #simulation or not) and thus is still naive, should be thus converted back to value like primed birds. In the data we will get some resistant 
            #birds withvuleof beta higher than it should, but these will never be used by the model, as these birds will never be infected again (since
            #already got 2 infections.
            elif self.state == 'RESISTANT':
                nbr_days_resistant = self.count_consecutive_x_at_end(li=self.li_states, x='R')
                #Note that below It is enough since in a resistant state, each time there was an E, there also was in I
                nbr_timeinfected = self.count_sequences_of_x(li=self.li_states, x='I') + self.experience2diseasepriormigration 
                if (nbr_days_resistant >= self.model.days_resistant) and (nbr_timeinfected<2):
                    self.state = 'SUSCEPTIBLE' 
                    #Note that in our case, this is gonna happen only when experience2diseasepriormigration=0 & 
                    #count_sequences_of_x(li=self.li_states, x='I')=1
                    #that is, only naive bird that got infected once will become again susceptible. In that case we also have to change its transmission
                    #and shedding rate, as it is no longer a naive individual.
                    self.beta_flbf = self.beta_flbf/self.model.excess_TR_naive
                    self.beta_ebf = self.beta_flbf/self.model.excess_TR_naive
                    self.beta_bb = self.beta_flbf/self.model.excess_TR_naive
                    self.excess_CO = 1
                    self.shedding_rate = self.model.shedding_rate
            
            #lastly, if the bird is susceptible, then check if it got infected
            elif self.state == 'SUSCEPTIBLE':
                #if the birds still have some distance to fly, it means that last step it was flying the whole day and thus 
                #cant get infected now, which mean it must keep the same state and dont do anything(SUSCEPTIBLE).
                #Otherwise, compute the different probability associated to the different pathways
                if self.remaining_distance2fly==0:
                    
                    ################################################################################################################
                    ######### compute proba of infection while RESTING: by virus in the env & (resting & exploring) birds ##########
                    ################################################################################################################
                    ### get info needed (which were registered at the end of the last step)
                    dico_nodeid_liRinfectedbird = self.model.dico_nodeid_liRinfectedbird
                    dico_nodeid_liEinfectedbird = self.model.dico_nodeid_liEinfectedbird 
                    dico_nodeid_effectivenbrbird = self.model.dico_nodeid_effectivenbrbird 
                    dico_birdid_info = self.model.dico_birdid_info 
                    dico_poultryid_info = self.model.dico_poultryid_info
                    beta_eb = self.model.beta_eb
                    virusinenv = self.model.dico_step_nodeid_virusinenv[self.step_id-1][self.pos]
                    #volume_site = self.model.G.nodes[self.pos]['W']*10**6
                    #control here the volume of water according to N so that we get only few simulationswith high prevalence
                    volume_site = self.model.water_volume_sites #in km³
            
                    
                    ### by virus in the environment
                    p_r_virusenv = self.proptimerestingtoday*(beta_eb*self.excess_CO)*virusinenv/volume_site
                 
                    ### by resting infected bird in same site (if any)
                    p_r_rbird = self.proptimerestingtoday*sum([1/dico_nodeid_effectivenbrbird[self.pos]*(dico_birdid_info[agent]['beta_bb']*self.excess_CO)*dico_birdid_info[agent]['proptimerestingtoday'] for agent in dico_nodeid_liRinfectedbird.get(self.pos,[])])

                    ### by birds infected birds that are resting at a site from which they can access the self.pos site during their
                    #explorative behaviour (if any) 
                    p_r_ebird = self.proptimerestingtoday*sum([1/dico_nodeid_effectivenbrbird[self.pos]*(dico_birdid_info[agent]['beta_bb']*self.excess_CO)*dico_birdid_info[agent]['proptimeexploring1node'] for agent in dico_nodeid_liEinfectedbird.get(self.pos,[])])

                    ### save info Needed to Compute the Proba (NCP)
                    if self.model.debuging:
                        self.decision_conditions['NCP_dico_nodeid_liRinfectedbird'] = dico_nodeid_liRinfectedbird
                        self.decision_conditions['NCP_dico_nodeid_liEinfectedbird'] = dico_nodeid_liEinfectedbird
                        self.decision_conditions['NCP_dico_birdid_info'] = dico_birdid_info
                        self.decision_conditions['NCP_dico_poultryid_info'] = dico_poultryid_info
                        self.decision_conditions['NCP_dico_nodeid_effectivenbrbird'] = dico_nodeid_effectivenbrbird
                        self.decision_conditions['NCP_virusinenv'] = virusinenv
                        self.decision_conditions['NCP_volume_site'] = volume_site
                        self.decision_conditions['NCP_model_beta_eb'] = beta_eb
                        self.decision_conditions['NCP_excess_CO'] = self.excess_CO


                    ############################################################################################################
                    ############### compute proba of getting infected while EXPLORING (if any sites to explore) ################
                    ############################################################################################################
                    #S_nf_*: any neighbouring sites/nodes reachable during foraging behaviour 
                    if self.model.debuging:
                        self.decision_conditions['NCP_S_nf_sites'] = self.S_nf_sites
                        self.decision_conditions['NCP_S_nf_farms'] = self.S_nf_farms
                    p_e_virusenv = 0
                    p_e_rbird = 0
                    p_e_ebird = 0

                    ##############################
                    ##### by exploring sites #####
                    ##############################
                    if len(self.S_nf_sites)>0:                                            
                                
                        ### by viruses in the environment of infected sites (as given at the end of the prior step)
                        p_e_virusenv = self.proptimeexploring1node*sum([(beta_eb*self.excess_CO)*\
self.model.dico_step_nodeid_virusinenv[self.step_id-1][nodeid]/(self.model.water_volume_sites) for nodeid in self.S_nf_sites])

                        ### by other resting birds in sites
                        p_e_rbird = self.proptimeexploring1node*sum([sum([1/dico_nodeid_effectivenbrbird[nodeid]*(dico_birdid_info[agent]['beta_bb']*self.excess_CO)*dico_birdid_info[agent]['proptimerestingtoday'] for agent in dico_nodeid_liRinfectedbird.get(nodeid,[])]) for nodeid in self.S_nf_sites])

                        ### by other exploring birds in sites or farms
                        p_e_ebird = self.proptimeexploring1node*sum([sum([1/dico_nodeid_effectivenbrbird[nodeid]*(dico_birdid_info[agent]['beta_bb']*self.excess_CO)*dico_birdid_info[agent]['proptimeexploring1node'] for agent in dico_nodeid_liEinfectedbird.get(nodeid,[])]) for nodeid in self.S_nf_sites+self.S_nf_farms])                      

                        
                    ##############################   
                    ##### by exploring farms #####
                    ##############################
                    p_e_farms = 0
                    if len(self.S_nf_farms)>0:
                        p_e_farms = self.proptimeexploring1node*sum([sum([dico_poultryid_info[agent]['beta_fb']*self.excess_CO for \
                                                                          agent in self.model.grid.get_cell_list_contents([nodeid]) if\
                                                                          (agent in dico_poultryid_info)]) for nodeid in self.S_nf_farms])                        

                    ############################################################################################################
                    ############## apply these proba in a random sequence, with the order reshuffled at each step ##############
                    ############################################################################################################
                    #list of proba the bird can become infected
                    dico_probaname_proba = {'p_it':self.model.p_it, 'p_r_rbird':p_r_rbird, 'p_r_ebird':p_r_ebird, 
                                            'p_r_virusenv':p_r_virusenv, 'p_e_rbird':p_e_rbird, 'p_e_ebird':p_e_ebird,
                                            'p_e_virusenv':p_e_virusenv, 'p_e_farms':p_e_farms}
                    for proba_name,proba_ in dico_probaname_proba.items():
                        self.decision_conditions[proba_name] = proba_

                    li_random_keys = list(dico_probaname_proba.keys())
                    random.shuffle(li_random_keys)
                    if self.model.debuging:
                        self.decision_conditions['li_random_keys'] = li_random_keys
                        
                    #iterate over the different transmission pathway
                    for proba_name in li_random_keys:
                        proba_ = dico_probaname_proba[proba_name]
                        if self.random.random()<proba_:
                            self.state = 'EXPOSED'

                            ########## keep track of the contamination pathway
                            self.dico_stepid_contaminationpathway[self.step_id] = proba_name
                            self.decision_conditions['contaminationpathway'] = proba_name

                            #break: dont try to infect further
                            break
            else:
                raise RuntimeError('ERROR: unkown state in Birds agent')
                #print('ERROR: unkown state in Birds agent')
                #sys.exit()
            
        #keep track of states
        self.li_states.append(self.state[0])
        
        
    def Behaviour(self):        
        
        '''migratory movements'''
        
        ######################################################################################################
        ######### then decide if the bird wants to migrate or not & update its body mass accordingly #########
        ######################################################################################################
        
        #add conditions for decision making for the migratory movements (since some birds could have migrated between
        #when the bird updated its state and when the bird decide to migrate or rest/explore)
        #store the location of birds at the time of performing the behaviour
        
        #save position of each agent for debuging
        if self.model.debuging:
            agents_on_node = self.model.grid.get_cell_list_contents([self.pos])
            dico_birdid_pos = {}
            for agent in agents_on_node:
                dico_birdid_pos[agent.unique_id] = agent.pos
            if self.model.debuging:
                self.decision_conditions['agentsonnode_4migratorydecision'] = dico_birdid_pos
        
        #if not dead
        if (self.state != 'DEAD'):
            
            ########### get info on current node & its nodetype
            current_node = self.pos
            nodetype_currentnode = self.model.G.nodes[current_node]["nodetype"]
            
            ########### if the bird is not flying the full day, lets just double check its node has a positive attractiveness
            #we compute the attractiveness of where the bird is now, or, if the bird is still flying, lets take the node 
            #attractiveness of where he will be landing, since he may be landing during the day, and thus we would still 
            #need to add some body mass based on this attractiveness.
            #That is, in this "get_node_attractiveness" function, when source_node_id==node_id, then we compute: 
            #days_flying = self.remaining_distance2fly/self.f, meaning that we will account for how long the birds still
            #need to fly for (could be 0 too)
            A_currentnode = self.get_node_attractiveness(step_id=self.step_id, source_node_id=current_node, node_id=current_node)
            if self.remaining_distance2fly<=self.f:
                if A_currentnode==0:
                    print('ERROR: current node has no food!')
                    print('A_currentnode: %.2f, current_node: %.2f, step_id: %.2f, DOY: %.2f, ice_end: %.2f, ice_start: %.2f, \
                           mean_foodavailability: %.2f' % (A_currentnode,current_node,self.step_id, self.model.DOY,
                                                  self.model.G.nodes[current_node]["ice_end"], 
                                                  self.model.G.nodes[current_node]["ice_start"],
                                                  self.model.G.nodes[current_node]["mean_foodavailability"]))
                    self.get_node_attractiveness(step_id=self.step_id, source_node_id=current_node, node_id=current_node, debug=True)
                    raise RuntimeError('ERROR, we are stoping the code')
                    #sys.exit()

            ########### if already in a migratory movement, then just update parameters and bodymass, and continue the route
            if self.remaining_distance2fly>0:
                
                ## D in km: the distance a bird flies over the day, computed as the minimum of the two: 
                #(1) the remaining distance to the new stopover site s_2 from the previous site s_1
                #(2) the maximum possible distance it could cover within a day at its flight speed
                D = min(self.remaining_distance2fly, self.f)
                #This is because the distance its available body fat could sustain is already accounted for when selecting the next site
                #but lets double check this: 
                #self.m-self.mlean: body fat
                #remove the cases when the bird was exposed and then got infected during flight but have not enough ressource since this
                #can happen
                if (self.remaining_distance2fly>(self.m-self.mlean)/self.rc_kgperkm)&\
                   (~((self.li_states[-1]=='I')&(self.li_states[-2]=='E'))):
                    print('-------------------')
                    print('self.remaining_distance2fly ',self.remaining_distance2fly)
                    print('self.m-self.mlean', self.m-self.mlean)
                    print('self.rc_kgperkm ', self.rc_kgperkm)
                    print('li_states[-5:] ', self.li_states[-5:])
                    print('ERROR: the bird is migrating to a site that is too far for its body mass to sustain!')
                    raise RuntimeError('ERROR, we are sotping the code')
                    #sys.exit()
                
                ## proptimenotflyingtoday: proportion of the day the bird will remain at the stopover (1- proportion of the day flying)
                self.proptimenotflyingtoday = 1-D/self.f #D in km, f in km/day --> km/(km/day)
                
                ## update body mass and remaining distance to fly
                self.update_bodymass(PTNF=self.proptimenotflyingtoday, A=A_currentnode, D=D)
                self.remaining_distance2fly = max(0,self.remaining_distance2fly-self.f)
               
                ## update li_pos now, not later (end of loop), since we need this info for "checkInfectionBelowMigratoryRoute"
                self.li_pos.append(current_node)
                self.li_remaining_distance2fly.append(self.remaining_distance2fly)
                                
                ##### assess whether there is any farms below its migratory movement (only if the bird is infected)
                if self.state == 'INFECTED':
                    self.checkInfectionBelowMigratoryRoute()
            
            
            ########### if not in a migratory movement then see if it can migrate
            else: 
                
                ########### Find all 'potential nodes' that can be used for migratory movement and for which the body fat can sustain 
                #the flight
                # and that are connected with orientation > 0 to the current node
                min_displacementwithoutrest = self.model.G.graph['min_displacementwithoutrest']
                km_sustained = (self.m-self.mlean)/self.rc_kgperkm
                li_nodes4mvt = [neighbor for neighbor in self.model.G.neighbors(current_node) if\
                                (self.model.G.edges[(current_node,neighbor)]["orientation"] > 0) and\
                                (km_sustained>=self.model.G.get_edge_data(current_node, neighbor)["min_distance2patch"]*10**3) and \
                                (self.model.G.get_edge_data(current_node, neighbor)["min_distance2patch"]>=min_displacementwithoutrest)]
                
                ########### Get a list of the potential nodes' attractiveness score (note that this has to be done for each bird at each
                #step separately, since bird density can vary within a same step across bird (e.g., the last bird activated will be less 
                #likely to move to the site with greater mean_foodavailability due to higher bird density))
                dico_neighbournodeID_A = {nodeid:round(self.get_node_attractiveness(step_id=self.step_id, 
                                                                              source_node_id=current_node, 
                                                                              node_id=nodeid),1) for nodeid in li_nodes4mvt}
                
                ########### Get a list of potential nodes' orientation score
                dico_neighbournodeID_O = {nodeid:self.model.G.get_edge_data(current_node, nodeid)["orientation"] for \
                                          nodeid in li_nodes4mvt}
                
                ########### Save info for debuging
                if self.model.debuging:
                    self.decision_conditions['li_nodes4mvt'] = li_nodes4mvt
                    self.decision_conditions['dico_neighbournodeID_A'] = dico_neighbournodeID_A
                    self.decision_conditions['dico_neighbournodeID_O'] = dico_neighbournodeID_O
                    #self.decision_conditions['A_bestOsite'] = A_bestOsite
                
                ########### If none of the plausible node has an attractivess>0 (i..e all covered by ice) OR if already in settlementsite,
                #then continue resting
                if (sum(dico_neighbournodeID_A.values())==0) or (self.model.G.nodes[self.pos]["nodetype"]=='settlementsite'):
                    self.proptimenotflyingtoday = 1 #set back to 1, in case the bird did not rest full day on the prior step
                    self.update_bodymass(PTNF=1, A=A_currentnode, D=0)
                    self.li_pos.append(current_node)
                    self.li_remaining_distance2fly.append(self.remaining_distance2fly)
                    #remaining_distance2fly is already = 0, no need to update.
                        
                ########### Else consider migrating
                elif sum(dico_neighbournodeID_A.values())>0:
                    
                    ########### compute probability of migrating
                    #if not yet changed site then migrate with a certain predefined proba, unless the bird has a delay (psi>0) (i.e., if
                    #infected, but coudl also be easily extended to also depend on if its a female/male, juvenile/adult)
                    if len(set(self.li_pos))==1:
                        proba_migrating = 0
                        if self.step_id>=self.psi:
                            proba_migrating = self.proba_startmigration
                            
                    #otherwise if already changed site       
                    else:
                        #keep only those position when the bird was not flying the entire day
                        li_pos = [str(int(pos)) for pos, dist in zip(self.li_pos, self.li_remaining_distance2fly) if dist == 0]
                        #print('-------------') #TODO: double check they are not always the same!
                        #print(li_pos)
                        #print(self.li_pos)
                        observed_MDOS = self.mean_consecutive_runs_exceptfirst(li_pos)
                        if self.model.debuging:
                            self.decision_conditions['observed_MDOS'] = observed_MDOS
                        #omega >0 for infected birds with delays else 0
                        exponent = (self.expected_MDOS-(observed_MDOS-self.expected_MDOS*self.omega))*self.model.temporal_synchrony
                        exponent_clipped = np.clip(exponent, -700, 700)  # Prevent overflow
                        proba_migrating = 1/(1+np.exp(exponent_clipped))

                    if self.model.debuging:
                        self.decision_conditions['proba_migrating'] = proba_migrating

                    ########### if the bird MIGRATE then update location based on its 'migration capacity'
                    new_node = current_node
                    if self.random.random() < proba_migrating:
                        
                        ###################################
                        ######### update location #########
                        ###################################
                        #### compute 'migration capacity' 
                        T = self.tradeoff
                        dico_neighbournodeID_MC = {n: dico_neighbournodeID_O[n]*T*(1 if A>0 else 0) + A*(1-T) for n, A in\
                                                   dico_neighbournodeID_A.items()}

                        ################### compute probabilities of choosing a site proportionally to its 'migration capacity'
                        nodeids = list(dico_neighbournodeID_MC.keys())
                        MP = np.array([dico_neighbournodeID_MC[n] for n in nodeids])
                        probabilities = MP / MP.sum() #so that the sum of all these proba sums to 1

                        #### adjust probabilities with respect to spatial_synchrony, where higher spatial_synchrony values 
                        # increase the likelihood of selecting the node with the highest probability. 

                        #First, create a probability distribution for when spatial synchrony=1. i.e. when the node with highest 
                        #proba is always selected
                        synch_proba = np.zeros_like(probabilities) #list of same length as probabilitites but only with 0.
                        synch_proba[np.argmax(probabilities)] = 1.0

                        # Then adjust the original and new distributions based on spatial_synchrony:
                        #when spatial_synchrony = 0, then adjusted_probabilities = np.array(probabilities)
                        #when spatial_synchrony = 0.5, then adjusted_probabilities = 0.5* np.array(probabilities) + 0.5* synch_proba
                        #when spatial_synchrony = 1, then adjusted_probabilities = synch_proba: i..e all use the best place
                        adjusted_probabilities = (1 - self.model.spatial_synchrony) * np.array(probabilities) + \
                                                 self.model.spatial_synchrony * synch_proba

                        # Normalize to ensure the probabilities sum to 1
                        adjusted_probabilities /= adjusted_probabilities.sum()  #so that the sum of all these proba sums to 1

                        # Choose a node with proba proportional to the adjusted probabilities
                        try:
                            new_node = np.random.choice(nodeids, p=adjusted_probabilities)
                        except Exception as e:
                            print(e)
                            print('dico_neighbournodeID_MC ',dico_neighbournodeID_MC)
                            print('dico_neighbournodeID_A ',dico_neighbournodeID_A)
                            print('dico_neighbournodeID_O ',dico_neighbournodeID_O)
                            print('T ',T)
                            print('spatial_synchrony ',self.model.spatial_synchrony)
                            raise RuntimeError('ERROR, we are sotping the code')
                            #sys.exit()

                        # save info for debuging
                        if self.model.debuging:
                            self.decision_conditions['dico_neighbournodeID_MC'] = {n:round(v, 3) for n,v in dico_neighbournodeID_MC.items()}
                            self.decision_conditions['dico_neighbournodeID_probaMC'] = dict(zip(nodeids, 
                                                                                      [round(v, 3) for v in adjusted_probabilities]))

                        self.last_edge_travelled = str(current_node)+'_'+str(new_node)

                    # move to new node (which could also equal current node)
                    self.model.grid.move_agent(self, new_node)  
                    self.li_pos.append(new_node)
                    
                    ###################################
                    #### update body mass & param. ####
                    ###################################
                    #if migrating
                    if current_node!=new_node:

                        #find distance2fly to reach new site when bird is migrating
                        Distance2fly = self.model.G.get_edge_data(current_node, new_node)['min_distance2patch']*10**3
                        #update the param accounting for whether if the bird needs >= a day to fly or not
                        if Distance2fly>self.f:
                            self.remaining_distance2fly = Distance2fly-self.f
                            self.proptimenotflyingtoday = 0
                        else:
                            self.remaining_distance2fly = 0 #not needed since should already be so but just to be clear
                            self.proptimenotflyingtoday = 1-Distance2fly/self.f

                        ##### see if there is any farms below (only if the bird is infected)
                        if self.state == 'INFECTED':
                            self.checkInfectionBelowMigratoryRoute()

                    #otherwise, if resting, then update birds attributes accordingly
                    else:
                        Distance2fly = 0
                        self.remaining_distance2fly = 0 #no need since should already be so but just to be clear
                        self.proptimenotflyingtoday = 1 #since at the begining its remaining distance to fly was == 0
                    
                    self.li_remaining_distance2fly.append(self.remaining_distance2fly)
                        
                    ###### keep some info for debuging
                    dico_neighbourANDselfnodeID_A = dico_neighbournodeID_A.copy()
                    dico_neighbourANDselfnodeID_A[current_node] = A_currentnode
                    if self.model.debuging:
                        self.decision_conditions['dico_neighbourANDselfnodeID_A'] = dico_neighbourANDselfnodeID_A

                    ###### update body mass
                    self.update_bodymass(PTNF=self.proptimenotflyingtoday, 
                                         A=dico_neighbourANDselfnodeID_A[new_node], 
                                         D=min(Distance2fly, self.f))
                    
                else:
                    raise RuntimeError('WEIRD ERROR, we are sotping the code')
                    #sys.exit()



###############################################################################################################################
########################################################## poultry ############################################################
###############################################################################################################################
class PoultryAgent(mesa.Agent):
    """
    Poultry Agent definition and its properties/interaction methods
    """

    def __init__(
        self,
        unique_id,
        model,
        initial_state,
        beta_fb,
        li_states,
        libetaflbf_of_Birds_passed_above,
        farmtype, 
        density, 
        number, 
        contaminationpathway,
        min_x,
        max_x,
        min_y,
        max_y,
        lambda0,
        xi_probadetectfarm
    ):
        super().__init__(unique_id, model)

        self.state = initial_state
        self.beta_fb = beta_fb
        self.li_states = li_states
        self.libetaflbf_of_Birds_passed_above = libetaflbf_of_Birds_passed_above
        self.farmtype = farmtype
        self.density = density
        self.number = number
        self.contaminationpathway = contaminationpathway
        self.min_x = min_x
        self.max_x = max_x
        self.min_y = min_y
        self.max_y = max_y
        self.lambda0 = lambda0
        self.xi_probadetectfarm = xi_probadetectfarm
    
    def count_consecutive_x_at_end(self, li, x):
            return sum(1 for _ in takewhile(lambda item: item == x, reversed(li)))
    
    def Behaviour(self):
        '''no behaviour for the farm'''
        pass
    
    def get_condition_snapshot(self):
        """Collect information on followed birds and their respective nodes"""
        # Get all attributes of the agent as a dictionary (except 'model' to avoid computational load)
        condition_snapshot = vars(self).copy()
        del condition_snapshot['model']
        # Retrieve current node information
        current_node = self.pos
        condition_snapshot['nodeid'] = current_node
        return condition_snapshot
    
    def Diseasedynamic(self):
        '''assess if the farm gets infected, and if so then see when the farmers detects it. Once it is detected (due to abnormal mortality
        for example, then cull the farm. While farmers prioritize depopulation, cleaning, and disinfection to restock poultry quickly, the
        process can take several months, so we do not replace culled farms'''
        
        # Track conditions (both its state and the site caracteristics) before any behavioural action
        if self.model.debuging:
            self.decision_conditions = self.get_condition_snapshot()    
        else: 
            self.decision_conditions = {}

        
        if self.step_id!=0:
            
            # If the farm had already been culled, then do nothing
            if self.state == 'DEAD':
                pass
            
            elif self.state == 'EXPOSED':
                #if still not enough days Exposed, then stay Exposed (do nothing), else check convert the birds to infected (i.e. start beeing infectious)
                nbr_days_withexposed = self.count_consecutive_x_at_end(li=self.li_states, x='E')
                if (nbr_days_withexposed >= self.model.days_exposed):
                    self.state = 'INFECTED'
                    
                # Compute proba of detecting something abnormal based on the number of days with symptoms (i.e.#days after incubation period)
                # If a virus is detected then culled, otherwise continue tracking its state              
                D = self.model.days_exposed-self.model.days_incubation #number of days since having symptoms
                if D>0:
                    proba_detectingAI = 1-np.exp(-self.lambda0*np.exp(self.xi_probadetectfarm*D))   
                    if self.random.random() < proba_detectingAI:
                        self.state = 'DEAD'
                        # remove it dynamically. The advantage of having dead farm removed from the grid is to not have it in 
                        # schedule anymore and thus gain computational time.
                        #self.model.grid.remove_agent(self) # Remove from the grid 
                        #self.model.schedule.remove(self)   # Remove from scheduler 
                        #self.remove()
                        
            # If the farm is infected then account for the fact that it is checked daily by the farmer attempting to detect 
            # anything abnormal (e.g., number of death). If the faremer detects the virus, the farm is culled
            elif self.state == 'INFECTED':
                # Compute proba of detecting something abnormal based on the number of days with symptoms (i.e.#days after incubation period)
                # If a virus is detected then culled, otherwise continue tracking its state
                #D: number of days since having symptoms
                D = (self.model.days_exposed+self.count_consecutive_x_at_end(li=self.li_states, x='I'))-self.model.days_incubation 
                if D>0:
                    proba_detectingAI = 1-np.exp(-self.lambda0*np.exp(self.xi_probadetectfarm*D))   
                    if self.random.random() < proba_detectingAI:
                        self.state = 'DEAD'
                        # remove it dynamically. The advantage of having dead farm removed from the grid is to not have it in 
                        # schedule anymore and thus gain computational time.
                        #self.model.grid.remove_agent(self) # Remove from the grid 
                        #self.model.schedule.remove(self)   # Remove from scheduler 
                        #self.remove()

            # If the farm is susceptible then update its state, i.e., see if it got infected
            elif self.state == 'SUSCEPTIBLE':

                ##### can get infected by birds flying above
                p_f_flbird = min(1,sum(self.libetaflbf_of_Birds_passed_above))

                ##### by birds exploring surrounding areas
                p_f_ebird = sum([self.model.dico_birdid_info[agent]['beta_ebf']/self.model.sanitation*\
                                 self.model.dico_birdid_info[agent]['proptimeexploring1node'] for agent in\
                                 self.model.dico_nodeid_liEinfectedbird.get(self.pos,[])])
                
                ##### by virus flowing via water
                # Compute new virus flowing into the node from connected water bodies
                #edge[0] is the source node, not self.pos
                NVW = sum(self.model.G.edges[edge]['ck'] * ((1-self.model.G.nodes[edge[0]]['decayrate'])**self.model.G.edges[edge]['T'])*\
                          self.model.dico_step_nodeid_virusinenv[self.step_id - self.model.G.edges[edge]['T']][edge[0]] for edge in\
                          self.model.G.in_edges(self.pos) if ('site' in self.model.G.nodes[edge[0]]['nodetype'])&\
                          (self.model.G.edges[edge]['waterspeed']>0)&(self.step_id >= self.model.G.edges[edge]['T']))
                p_f_virusenv = self.model.beta_vf * NVW / self.model.sanitation
                if self.model.debuging:
                    self.decision_conditions['newvirusfrom_flowingINwaterbodies'] = NVW
                
                ##### by secondary spread from connected infected farms TODO for now we just have a certain risk of 
                # secondary infection as long as the farms are connected
                #we use dico_poultryid_info here instead of like agent.state because we want the agent state as given at the end of the 
                #prior step
                S_connectedinfectedfarms = [sum([1 for agent in self.model.grid.get_cell_list_contents([n]) if\
                                                 agent.unique_id in self.model.dico_poultryid_info]) for\
                                            n in self.model.G.neighbors(self.pos) if ('farm' in self.model.G.nodes[n]['nodetype'])]            
                p_f_f = self.model.beta_ff*sum(S_connectedinfectedfarms)/self.model.sanitation
                if self.model.debuging:
                    self.decision_conditions['S_connectedinfectedfarms'] = S_connectedinfectedfarms
                
                ##### apply these proba in a random sequence, with the order reshuffled at each step
                #list of proba the bird can become infected
                dico_probaname_proba = {'p_f_flbird':p_f_flbird, 'p_f_ebird':p_f_ebird, 'p_f_virusenv':p_f_virusenv, 'p_f_f':p_f_f}
                for proba_name,proba_ in dico_probaname_proba.items():
                    #if self.model.debuging: #quite usefull in non-debugging mode to
                    self.decision_conditions[proba_name] = proba_

                li_random_keys = list(dico_probaname_proba.keys())
                random.shuffle(li_random_keys)
                if self.model.debuging:
                    self.decision_conditions['li_random_keys'] = li_random_keys
                    
                #iterate over the different transmission pathways
                for proba_name in li_random_keys:
                    proba_ = dico_probaname_proba[proba_name]
                    
                    #if this pathway infect the bird we break the loop otherwise continue to interate
                    if self.random.random()<proba_:
                        self.state = 'EXPOSED'

                        ########## keep track of the contamination pathway
                        self.decision_conditions['contaminationpathway'] = proba_name

                        #break: dont try to infect further
                        break

            else:
                #print('ERROR: unknown state in poultry agent')
                raise RuntimeError('ERROR: unknown state in poultry agent')
                #sys.exit()
        
        #keep track of states
        self.li_states.append(self.state[0])
        #reset it to 0
        self.libetaflbf_of_Birds_passed_above = []



