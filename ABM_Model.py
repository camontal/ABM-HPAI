import numpy as np
import random
import os
import sys
import pickle
import math

import mesa
from itertools import takewhile, groupby
import networkx as nx
import matplotlib.pyplot as plt
from shapely.geometry import LineString, Polygon
from scipy.stats import exponnorm

from ABM_Agents import BirdsAgent, PoultryAgent
from ABM_DataCollection import followedagent_info, model_info

sys.path.append('C:\\Users\\montalci\\Desktop\\AImigration\\G1')
import config



#adapted from: https://github.com/projectmesa/mesa-examples/blob/main/examples/virus_on_network/virus_on_network/model.py
class MigrationandVirus(mesa.Model):
    """
    A virus model with some bird and farm agents
    """
    def __init__(self, GID, N, checkice, initial_outbreak_proportion, initial_immune_proportion, 
                 days_infected, days_exposed, days_resistant, days_incubation, days_symptom,
                 beta_bb_mean, beta_flbf_mean, beta_ebf_mean, beta_bird_sd, 
                 beta_fb, beta_vf, beta_ff, beta_eb, 
                 body_mass_mean, body_mass_sd, theta, f_ms, 
                 prop_naive, excess_CO_naive, excess_TR_naive, excess_SH_naive,
                 nbr_birds2follow, tradeoff_mean, tradeoff_sd, 
                 DOY, SYM, expected_MDOS, proba_startmigration, temporal_synchrony,
                 spatial_synchrony, birddensity_scaling, 
                 shedding_rate, deadbirdscalingshedding, p_it,base_transmission_rate,
                 generalproptimeresting_mean, generalproptimeresting_sd,
                 home_range_area_I, home_range_area_I_scalenaive, generalproptimeresting_I, generalproptimeresting_I_scalenaive, 
                 omega, omega_scalenaive, proba_dying, proba_dying_scalenaive,
                 delayedmigrationinfectedLOC, delayedmigrationinfectedLOC_scalenaive, 
                 lambda0chicken, lambda0duck, xi_probadetectfarm, sanitation,
                 alpha_T, gamma_T, alpha_pH, gamma_pH, watertemperature, pH, ck, water_volume_sites, #PW_beta, PW_watertemperature,
                 dico_stepid_nodeID_attractiveness=[{}], printdebug=False, debuging=False):
        
        '''
        DOY: day of year that correspond to the first time step
        '''
        
        super().__init__()
        
        ################################################
        ############### define scheduler ###############
        ################################################
        #Initialize a schedulers, i.e., an object which controls when agents are called upon to act, and when.
        #p.54: chrome-extension://efaidnbmnnnibpcajpcglclefindmkaj/https://mesa.readthedocs.io/_/downloads/en/main/pdf/
        #shuffle: Determines whether to shuffle the order of agents each step.
        #the scheduler allows agent activation to be divided into several stages instead of a single `step` method. 
        #All agents execute one stage before moving on to the next. Agents must have all the stage methods implemented,
        #which takes a model object as only argument
        self.schedule = mesa.time.StagedActivation(self, stage_list=['Diseasedynamic','Behaviour'], shuffle=True)
        
        ################################################
        ################ parametrization ###############
        ################################################
        ########## open graph G open if its a graph else try to open a graph with this name, otherwise thorugh an error
        try:
            with open(os.path.join(config.path_save_networkspace, GID+ '_graph.pkl'), 'rb') as f:
                G = pickle.load(f)
        except Exception as e:
            print(e)
            raise RuntimeError('ERROR: your GID parameter is wrong!')
            
        ########## add decay rate
        #G.graph['alpha_T'] = alpha_T
        self.alpha_T = alpha_T
        self.gamma_T = gamma_T
        self.alpha_pH = alpha_pH
        self.gamma_pH = gamma_pH
        self.watertemperature = watertemperature
        self.pH = pH
        self.ck = ck
        self.water_volume_sites = water_volume_sites
        G = self.adddecayrateANDck(G=G)
        
        ########## save graph & other attributes
        self.G = G
        self.GID = GID #for debugging
        self.N = N
        self.grid = mesa.space.NetworkGrid(self.G)
        self.DOY = DOY
        self.printdebug = printdebug
        self.checkice = checkice
        self.debuging = debuging

        #related to symptoms
        self.omega = omega
        self.omega_scalenaive = omega_scalenaive
        self.proba_dying = proba_dying
        self.proba_dying_scalenaive = proba_dying_scalenaive
        self.delayedmigrationinfectedLOC = delayedmigrationinfectedLOC
        self.delayedmigrationinfectedLOC_scalenaive = delayedmigrationinfectedLOC_scalenaive
        
        # Store the attractiveness changes dictionary
        self.dico_stepid_nodeID_attractiveness = dico_stepid_nodeID_attractiveness

        #virus related
        self.initial_outbreak_proportion = initial_outbreak_proportion
        self.initial_immune_proportion = initial_immune_proportion
        self.days_infected = days_infected
        self.days_exposed = max(days_exposed,1) #have to be at least 1 day, we follow SEIRS
        self.days_incubation = max(days_incubation,1)
        self.days_resistant = days_resistant
        self.days_symptom = days_symptom
        self.shedding_rate = shedding_rate
        self.base_transmission_rate = base_transmission_rate
        self.dico_step_nodeid_virusinenv = {} #we need for prior steps because virus can take time to travel in rivers
        self.dico_step_nodeid_birdidINFO = {} #for consistency, to avoid confusion, we thus also keep this one like this
        self.deadbirdscalingshedding = deadbirdscalingshedding

        #pathway related
        #self.PW_beta = PW_beta
        #self.PW_watertemperature = PW_watertemperature
        
        #check proba are fine
        if self.initial_outbreak_proportion+self.initial_immune_proportion>1:
            raise RuntimeError('ERROR: you cant have initial_outbreak_proportion-initial_immune_proportion>1')
            #print('ERROR: you cant have initial_outbreak_proportion-initial_immune_proportion>1')
            #sys.exit()        
            
        #bird related
        self.beta_vf = beta_vf
        self.beta_ff = beta_ff
        self.beta_flbf_mean = beta_flbf_mean
        self.beta_ebf_mean = beta_ebf_mean
        self.beta_bb_mean = beta_bb_mean
        self.beta_eb = beta_eb
        self.beta_bird_sd = beta_bird_sd
        self.p_it = p_it
        self.beta_fb = beta_fb

        self.body_mass_mean = body_mass_mean
        self.body_mass_sd = body_mass_sd
        self.f_ms = f_ms
        self.theta = theta
        
        self.sanitation = sanitation

        self.prop_naive = prop_naive
        self.excess_TR_naive = excess_TR_naive
        self.excess_CO_naive = excess_CO_naive
        self.excess_SH_naive = excess_SH_naive
        self.nbr_birds2follow = nbr_birds2follow
        self.expected_MDOS = expected_MDOS
        self.tradeoff_mean = tradeoff_mean
        self.tradeoff_sd = tradeoff_sd
        self.proba_startmigration = proba_startmigration
        self.spatial_synchrony = spatial_synchrony
        self.temporal_synchrony = temporal_synchrony
        self.birddensity_scaling = birddensity_scaling
        
        self.generalproptimeresting_mean = generalproptimeresting_mean
        self.generalproptimeresting_sd = generalproptimeresting_sd
        self.generalproptimeresting_I = generalproptimeresting_I
        self.generalproptimeresting_I_scalenaive = generalproptimeresting_I_scalenaive
        self.home_range_area_I = home_range_area_I
        self.home_range_area_I_scalenaive = home_range_area_I_scalenaive
        
        #poultry related
        self.xi_probadetectfarm = xi_probadetectfarm
        self.lambda0chicken = lambda0chicken
        self.lambda0duck = lambda0duck

        #collect at each step for all iteration and run, the number of infected, susceptible and resistant agents (for all agent's type)
        #also collect where 20 random birds are
        self.datacollector = mesa.DataCollector(model_reporters={'followedagent_info':followedagent_info, 'model_info':model_info}) 

        
        ########################################################################################################
        ########################################### Add birds agents ###########################################
        ########################################################################################################

        #Compute M (in km) as the mean distance of all edge with orientation score =1. to compute rc_kgperkm later on
        M = np.mean([data['min_distance2patch'] for u, v, data in self.G.edges(data=True) if data['orientation']==1])*10**3
        #Fmin = np.min([data['mean_foodavailability'] for n, data in self.G.nodes(data=True) if data['mean_foodavailability']>0])

        ###### Extract nodes with 'startingsite' type and their area sizes
        dico_startingsite_squatermeter = {node: G.nodes[node]['area_size'] for node in G.nodes if\
                                          G.nodes[node]['nodetype']=='startingsite'}
        total_area = sum(dico_startingsite_squatermeter.values())   
        if total_area == 0:
            raise ValueError("Total area size of all sites must be greater than zero")
            
        ####### if a single population
        # Compute bird distribution based on proportional area size
        dico_node_distribution = {node: round((area / total_area) * N) for node, area in dico_startingsite_squatermeter.items()}

        # Adjust for rounding errors to ensure exactly N birds are distributed
        assigned_birds = sum(dico_node_distribution.values())
        difference = self.N - assigned_birds

        if difference != 0:
            #sort in descending order for fair repartition: list of nodes sorted by area size (largest first)
            sorted_nodes = sorted(dico_startingsite_squatermeter, key=lambda node: -dico_startingsite_squatermeter[node]) 
            for i in range(abs(difference)):
                node = sorted_nodes[i % len(sorted_nodes)]
                dico_node_distribution[node] += 1 if difference > 0 else -1
        if sum(dico_node_distribution.values())!=N:
            raise RuntimeError('ERROR: your birds are not distributed well in the starting nodes')
            #print('ERROR: your birds are not distributed well in the starting nodes')
            #sys.exit()

        ####### if several populations
        # put one population per startingsite? to do as possible extension
        
        ####### scaling the infection probability onto the non-immune subset 
        if not (0 <= initial_outbreak_proportion <= 1):
            raise ValueError("initial_outbreak_proportion must be in [0, 1].")
        if not (0 <= initial_immune_proportion <= 1):
            raise ValueError("initial_immune_proportion must be in [0, 1].")
        nbr_infectedbirds = self.N * self.initial_outbreak_proportion
        nbr_availablebird4infection = self.N * (1-self.initial_immune_proportion)
        if nbr_infectedbirds > nbr_availablebird4infection:
            raise RuntimeError('ERROR: you dont have enough birds to infect')
            #print('ERROR: you dont have enough birds to infect')
            #sys.exit()
        else:
            proba_gettinginfected = nbr_infectedbirds/nbr_availablebird4infection
            
        # Compute exposed proba probabilities
        p_exposed = 1 - (self.prop_naive + self.initial_immune_proportion)
        
        # Adjust if naive + immune > 1 (which would make p_exposed negative)
        if self.prop_naive + self.initial_immune_proportion > 1:
            correction = (self.prop_naive + self.initial_immune_proportion - 1) / 2.0
            self.prop_naive -= correction
            self.initial_immune_proportion -= correction
            p_exposed = 0  # no room for previously exposed group
         # Make sure probabilities sum to 1 
        li_probs = [self.prop_naive, p_exposed, self.initial_immune_proportion]
        li_probs = np.clip(li_probs, 0, 1)  # ensure none are negative
        li_probs = li_probs / np.sum(li_probs)            
        
        ####### Create N bird agents distributed equally across all starting sites (thats where they should start in our example)    
        bird_id = 0 # Initialize a unique ID counter for birds agents
        for node,nbrbird in dico_node_distribution.items():
            for _ in range(0, nbrbird):

                ### define the bird's state, whether it is naive, non-naive (susceptible or infected), or immune 
                #0: immunologically naive
                #1: got exposed once already long time ago, and is now non-naive, either in a susceptible of infected state
                #2: immune as the bird got infected twice
                experience2diseasepriormigration = np.random.choice([0, 1, 2], p=li_probs)
                
                #set if bird is infected or not, cosntraining to the non-immune birds (i.e. those that experience2diseasepriormigration!=2), but keeping
                #the same proportion as asked by the user
                init_s = ('RESISTANT' if experience2diseasepriormigration==2 else \
                         ('INFECTED' if (self.random.random() <= proba_gettinginfected) else 'SUSCEPTIBLE'))

                ### define the bird's general proportion of time spent resting vs exploring
                symptom = (1 if (self.random.random() <= SYM) else 0)
                hra = self.G.graph['max_explorativewithoutrest']*10**3
                #not that gpr_adjusted can be >1, the reason is that we want to be able to reverse to its original value when birds 
                #are no longer sick. This is why we will control for values >1 later on (in this notebook) like this: 
                #agent.proptimerestingtoday = agent.proptimenotflyingtoday*min(agent.generalproptimeresting,1)
                gpr = round(np.clip(np.random.normal(self.generalproptimeresting_mean, self.generalproptimeresting_sd),0.05,1), 3)
                gpr_adjusted = gpr*self.generalproptimeresting_I*self.generalproptimeresting_I_scalenaive if (init_s=='INFECTED' and symptom==1 and\
                                                                                                              experience2diseasepriormigration==0) else\
                               gpr*self.generalproptimeresting_I if (init_s=='INFECTED' and symptom==1 and experience2diseasepriormigration==1) else gpr
                ### initial body mass and theta
                m0 = np.clip(np.random.normal(self.body_mass_mean, self.body_mass_sd),0.15,14.5)
                #daily fuel deposition rate, in % of lean mass
                rd_perclean = 1.16*m0**-0.35
                #lean mass remaining constant throughout
                mlean = self.theta*m0
                #fuel consumption rate in kg/km 
                rc_kgperkm = (self.expected_MDOS * rd_perclean/100*mlean) / M
                if self.debuging:
                    if (bird_id==0) and (random.random() < 1/50):
                        print('\nA bird of initial body mass %.4f kg, '
                            'will consume %.10f kg/km, '
                            'and will fly on average %.4f km per migration route, '
                            'and thus will lose on average %.4f kg per migration route, '
                            'which will take %.4f days to regain, '
                            'since it can gain %.4f kg per day' %
                            (m0, rc_kgperkm, M, rc_kgperkm * M, rc_kgperkm * M / (rd_perclean / 100 * mlean), rd_perclean / 100 * mlean))
                #note that generalproptimeresting can be >1: we need this info for converitng it back to normal when the brid become immune again
                
                ### Heterogeneity in infectiousness
                # Means for each parameter
                means = [self.beta_flbf_mean, 
                         self.beta_ebf_mean, 
                         self.beta_bb_mean]
                # Standard deviations
                stds = [self.beta_flbf_mean * self.beta_bird_sd, 
                        self.beta_ebf_mean * self.beta_bird_sd,  
                        self.beta_bb_mean * self.beta_bird_sd]
                # Correlation coefficient (1.0 = perfectly correlated, 0 = independent)
                rho = 0.8  # adjust as needed
                # Correlation matrix between the transmission rate
                corr_matrix = np.array([ [1.0, rho, rho], [rho, 1.0, rho], [rho, rho, 1.0]])
                # Convert correlation matrix to covariance matrix
                cov_matrix = np.outer(stds, stds) * corr_matrix
                # Draw correlated values
                beta_flbf_init, beta_ebf_init, beta_bb_init = np.random.multivariate_normal(means, cov_matrix)

                ### delayed migration for infected birds with symptoms. naive birds get their value multiplied by a value>1
                #values should be between 0 and 50, but values of 50 are replaced by 300 to ensure brids dont migrate.
                loc = self.delayedmigrationinfectedLOC
                if loc != -1:
                    psi = (exponnorm.rvs(5, loc=loc, scale=1, size=1)[0] if init_s=='INFECTED' and symptom==1 else 1)
                    psi = max(psi, 1)
                    if experience2diseasepriormigration==0 and init_s=='INFECTED' and symptom==1:
                        psi = psi*self.delayedmigrationinfectedLOC_scalenaive
                    psi = 300 if psi >= 50 else psi
                else:
                    psi = 1
                a = BirdsAgent(
                    unique_id = 'bird'+str(bird_id), 
                    model = self,
                    initial_state = init_s,
                    symptom = symptom, #for infected and non-infected birds
                    proba_dying = self.proba_dying*self.proba_dying_scalenaive if experience2diseasepriormigration==0 and init_s=='INFECTED' and\
                                  symptom==1 else \
                                  self.proba_dying if experience2diseasepriormigration==1 and init_s=='INFECTED' and symptom==1 else 0,
                    omega = self.omega if experience2diseasepriormigration==0 and init_s=='INFECTED' and symptom==1 else \
                            self.omega*self.omega_scalenaive if experience2diseasepriormigration==1 and init_s=='INFECTED' and symptom==1 else 0,
                    daysalreadywithsymptomprior = random.randint(0, min(5,self.days_symptom-1)), #add a bit of variation
                    experience2diseasepriormigration = experience2diseasepriormigration,
                    beta_flbf = beta_flbf_init*self.excess_TR_naive if experience2diseasepriormigration==0 else beta_flbf_init,
                    beta_ebf = beta_ebf_init*self.excess_TR_naive if experience2diseasepriormigration==0 else beta_ebf_init,
                    beta_bb = beta_bb_init*self.excess_TR_naive if experience2diseasepriormigration==0 else beta_bb_init,
                    excess_CO = self.excess_CO_naive if experience2diseasepriormigration==0 else 1, #for now only naive can lead to an excess contraction
                    shedding_rate = self.shedding_rate*self.excess_SH_naive if experience2diseasepriormigration==0 else self.shedding_rate, 
                    home_range_area = hra*self.home_range_area_I if (init_s=='INFECTED' and symptom==1 and experience2diseasepriormigration==0) else\
                                      hra*self.home_range_area_I*self.home_range_area_I_scalenaive if (init_s=='INFECTED' and symptom==1 and \
                                                                                            experience2diseasepriormigration==1) else hra, 
                    m0 = m0,
                    mlean = mlean, #we assume that lean mass remains constant throughout the simulation
                    f = self.f_ms*86.4, #converted into km/day by *86.4.
                    rc_kgperkm = rc_kgperkm,
                    rd_perclean = rd_perclean,
                    generalproptimeresting = gpr_adjusted, 
                    remaining_distance2fly = 0,
                    proptimenotflyingtoday = 1,
                    proptimeexploring1node = 0, #will be: (agent.proptimenotflyingtoday*(1-agent.generalproptimeresting))/len(S_nf)
                    proptimerestingtoday = min(gpr_adjusted,1), #will be: agent.proptimenotflyingtoday*agent.generalproptimeresting
                    expected_MDOS = self.expected_MDOS,
                    li_states = [init_s[0]], 
                    li_pos = [node],
                    li_remaining_distance2fly = [0], #should include the value of the updated actual step as well
                    tradeoff = round(np.clip(np.random.normal(self.tradeoff_mean, self.tradeoff_sd), 0, 1),3), #ensure between 0-1,
                    proba_startmigration = proba_startmigration,
                    dico_stepid_contaminationpathway = {},
                    S_nf_sites = [],
                    S_nf_farms = [],
                    li_farmpassedabove = [],
                    psi = psi #note: should not change after teh birds is no longer infected otherwise it will never get bigger than the infectious period
                )

                
                #if the bird start in an infected state with symptom, then add the effect of AI on its physiology & behaviour
                if (init_s == 'INFECTED') & (a.symptom==1):
                    a.Symptomseffect(is_reverse=False)
                    
                # Add the agent to the schedule and node
                self.schedule.add(a)
                self.grid.place_agent(a, node)
                bird_id += 1  

        
        ############# select some random birdID to follow their movements through all the steps
        agents_on_all_nodes = self.grid.get_cell_list_contents(self.G.nodes())
        bird_agents = [obj for obj in agents_on_all_nodes if isinstance(obj, BirdsAgent)]
        self.birdid2follow = list(set(self.random.sample(bird_agents, min(self.nbr_birds2follow, len(bird_agents)))))
        #print([a.unique_id for a in self.birdid2follow])

        
        ########################################################################################################
        ########################################## Add poultry agents ##########################################
        ########################################################################################################      
        # Adding chickens and duck farms based on the network space. 
        # By construction, there is one chicken or one duck farms we set the attributes based on the node's properties (assuming the agent
        #is placed on a node with these properties)
        farm_id = 0 # Initialize a unique ID counter for birds agents
        for node in self.G.nodes():
            nodetype = self.G.nodes[node]['nodetype']
            if nodetype in ['duckfarm', 'chickenfarm']:
                a = PoultryAgent(
                    unique_id = 'farm'+str(farm_id), 
                    model = self,
                    initial_state = 'SUSCEPTIBLE',
                    beta_fb = self.beta_fb, #should it be as a function of population size?
                    li_states = ['S'],
                    libetaflbf_of_Birds_passed_above = [],
                    farmtype = nodetype,
                    density = self.G.nodes[node]["density"], #not using it for now, as not studying effect of farm heterogeneity on spread
                    number = self.G.nodes[node]["number"], #not using it for now, as not studying effect of farm heterogeneity on spread
                    contaminationpathway = None,
                    min_x = self.G.nodes[node]['min_x'], 
                    max_x = self.G.nodes[node]['max_x'], 
                    min_y = self.G.nodes[node]['min_y'], 
                    max_y = self.G.nodes[node]['max_y'],
                    lambda0 = (self.lambda0chicken if nodetype=='chickenfarm' else self.lambda0duck),
                    xi_probadetectfarm = self.xi_probadetectfarm
                )  
        
                self.schedule.add(a)
                # Add the agent to the node
                self.grid.place_agent(a, node)
                farm_id += 1    
                
        ############# also automatically follow all farms
        agents_on_all_nodes = self.grid.get_cell_list_contents(self.G.nodes())
        self.birdid2follow.extend([obj for obj in agents_on_all_nodes if isinstance(obj, PoultryAgent)])

        
        ########################################################################################################
        ####################################### initialise step counter ########################################
        ########################################################################################################              
        # initialize my own step counter (primarly for double checking) we start the day prior step 0, to save  
        # info about the birds for the last 24h where all birds staid resting at their starting site
        self.step_id = -1  
        self.update_virus_in_environment()   #we assume no virus in the environment at step_id = -1
        #self.saveinfo4diseasedynamic() #no need as there will be no infection dynamics at step 0
        
        #then add 1 to start at 0, like the step counter of mesa and add it to each agent
        self.step_id += 1
        for agent in self.schedule.agents:
            agent.step_id = self.step_id
        self.running = True
        
        
    def count_consecutive_x_at_end(self, li, x):
        return sum(1 for _ in takewhile(lambda item: item == x, reversed(li)))      
    #small examples:
    #li = ['I','S','S','S','I','I','D','D']
    #count_sequences_of_x(li=li, x='S') #1
    #count_sequences_of_x(li=li, x='I') #2
    #count_consecutive_x_at_end(li=li, x='I') #0
    #count_consecutive_x_at_end(li=li, x='D') #2     
    
    def adddecayrateANDck(self, G):
        """
        Updates each node in the graph with a 'decayrate' attribute based on its temperature (T) and pH.
        'watertemperature' (temperature) and 'pH' are given as paramter of the model
        alpha_T: Virus decay rate at 0°C
        gamma_T: Sensitivity of viral decay to temperature
        alpha_pH: Width of the bell curve for pH influence
        gamma_pH: Center of the peak for pH influence
        """
        # Temperature-dependent decay rate
        eta_T = self.alpha_T * math.exp(self.gamma_T * self.watertemperature)

        # pH adjustment factor
        v = 1.5 - math.exp(-0.5 * ((self.pH - self.gamma_pH) / self.alpha_pH) ** 2)

        # Final decay rate
        eta = eta_T * v
        
        # Assign decay rate to each site. it is assigned ot each node, so that we can also have variability per node by easily
        #modifying this code, e.e.g with real graph
        for node in G.nodes:
            if 'site' in G.nodes[node]['nodetype']:
                G.nodes[node]['decayrate'] = eta
                G.nodes[node]['ck'] = self.ck
        # Assign ck to edges where waterspeed > 0
        for u, v, data in G.edges(data=True):
            if data.get('waterspeed', 0) > 0:
                G.edges[u, v]['ck'] = self.ck
        return G
        
    def saveinfo4diseasedynamic(self):
        '''Save info related to bird agent: total effective number,  some characteristics of infected ones, and for each node a list
        of infected birds resting at the site, and those exploring the site'''
        
        ######################################################
        ######################## bird ########################
        ######################################################
        #to save necessary info for each infected birds
        dico_birdid_info = {}
        
        #to get the 'effective' number of birds including infected and non-infected birds (i.e., a bird spending half a day 
        #would be counted as 0.5) so it is not an integer in this site
        dico_nodeid_effectivenbrbird = {}
        #list of resting and epxloring birds for each node
        dico_nodeid_liRinfectedbird = {} #R stands for "Resting", nodeid stand for any node, site or farms
        dico_nodeid_liEinfectedbird = {} #E stands for "Exploring",  nodeid stand for any node, site, farms, or exploring
            
        # Iterate over all bird agent to save the info
        for agent in self.schedule.agents:
            
            # Update some attribute for all birds' agents
            if (isinstance(agent, BirdsAgent)):
                agent.S_nf_sites = []
                agent.S_nf_farms = []
                agent.proptimerestingtoday = agent.proptimenotflyingtoday*min(agent.generalproptimeresting,1)
                agent.proptimeexploring1node = 0
                
            # Save the info only if the agent is still flying as birds flying can't contribute to spreading the virus. 
            # We also dont account for dead birds, since they only contribute to contamination in the environment and not via direct 
            #bird contact
            # Here it is not only infected birds, since we need to compute the effective number of birds, which accounts for
            #any alive birds not migrating to another site            
            if (isinstance(agent, BirdsAgent)) and (agent.remaining_distance2fly==0) and (agent.state!='DEAD'):

                ######## extract any neighbouring nodes n (farm or sites) reachable during agents' explorative behaviour 
                #and save it as agents attribute
                
                #IF THE BIRD HAVE TIME FOR EXPLORING only, otherwise dividing by 0 in the p_e_ebird for example
                S_nf_sites, S_nf_farms = [], []
                if agent.generalproptimeresting<1:
                    for n in self.G.neighbors(agent.pos):
                        if self.G.edges[(agent.pos, n)]['min_distance2patch']*10**3 <= agent.home_range_area:
                            if self.G.nodes[n]['nodetype'] in ['duckfarm', 'chickenfarm']:
                                S_nf_farms.append(n)
                            elif self.G.nodes[n]['nodetype'] in ['settlementsite', 'stopoversite','startingsite','exploringsite']:
                                S_nf_sites.append(n)
                            else:
                                raise RuntimeError('ERROR: check your nodetype values')
                                #print('ERROR: check your nodetype values')
                                #sys.exit()
                #update the list of sites and farms that the bird can explore during this step
                agent.S_nf_sites = S_nf_sites
                agent.S_nf_farms = S_nf_farms
                
                ######## update agent's time spent exploring nodes if there are any sites, epxloring nodes, or farms to explore
                #total number of nodes that the bird can explore
                S_nf = S_nf_sites + S_nf_farms
                if len(S_nf)!=0:
                    agent.proptimeexploring1node = (agent.proptimenotflyingtoday*(1-min(agent.generalproptimeresting,1)))/len(S_nf)
            
                ######## add contribution to the effective number of birds in a node (i.e.,from both infected & non-infected birds)
                # Add the bird contribution to its RESTING site as a resting bird
                if agent.pos not in dico_nodeid_effectivenbrbird:
                    dico_nodeid_effectivenbrbird[agent.pos] = 0
                dico_nodeid_effectivenbrbird[agent.pos] += agent.proptimerestingtoday
                # Add the bird contribution to its EXPLORATIVE sites (if any) as an explorative bird
                for nodeid in S_nf:
                    if nodeid not in dico_nodeid_effectivenbrbird:
                        dico_nodeid_effectivenbrbird[nodeid] = 0                    
                    dico_nodeid_effectivenbrbird[nodeid] += agent.proptimeexploring1node
                
                ######## save the bird's info if it is infected as well as keep it in a list for all node that it is affecting
                if agent.state == 'INFECTED':
                    # Save the bird info that will be used to infect farms/birds
                    dico_birdid_info[agent.unique_id] = {'beta_bb': agent.beta_bb,
                                                         'beta_ebf': agent.beta_ebf,
                                                         'proptimeexploring1node': agent.proptimeexploring1node,
                                                         'proptimerestingtoday': agent.proptimerestingtoday,
                                                         'generalproptimeresting': min(agent.generalproptimeresting,1),
                                                         'shedding_rate': agent.shedding_rate}
                    # Add the bird contribution to its RESTING site as a resting bird
                    if agent.pos not in dico_nodeid_liRinfectedbird:
                        dico_nodeid_liRinfectedbird[agent.pos] = []
                    dico_nodeid_liRinfectedbird[agent.pos].append(agent.unique_id)
                    # Add the bird contribution to its EXPLORATIVE sites (if any) as an explorative bird
                    for nodeid in S_nf:
                        if nodeid not in dico_nodeid_liEinfectedbird:
                            dico_nodeid_liEinfectedbird[nodeid] = []   
                        dico_nodeid_liEinfectedbird[nodeid].append(agent.unique_id)     
                    
        #save the result in the model's attribute
        self.dico_nodeid_liRinfectedbird = dico_nodeid_liRinfectedbird 
        self.dico_nodeid_liEinfectedbird = dico_nodeid_liEinfectedbird 
        self.dico_nodeid_effectivenbrbird = dico_nodeid_effectivenbrbird
        self.dico_birdid_info = dico_birdid_info 
        
        ######################################################
        ######################## farm ########################
        ######################################################
        #save necessary info for each infected farms. (because we need the prior state of farm since farms can infect other farms and birds)
        dico_poultryid_info = {}
        for agent in self.schedule.agents:
            if (isinstance(agent, PoultryAgent)) and (agent.state=='INFECTED'):
                dico_poultryid_info[agent.unique_id] = {'beta_fb': agent.beta_fb}
        self.dico_poultryid_info = dico_poultryid_info 
            
            
    def update_virus_in_environment(self):   
        '''Compute the virus load in the environment based on infected birds.
        This function is executed at the end of each step to capture the birds' states during that step, i.e. after all agents have 
        been activated. 
        The resulting values are then used to determine new infections at the beginning of the next step.'''
        self.dico_step_nodeid_virusinenv[self.step_id] = {}
        self.dico_step_nodeid_birdidINFO[self.step_id] = {} #just to debug. could be removed in the last version for efficiency
        
        ##########################################################################################################################
        ###################################### sum the shedding rate of all infected birds #######################################
        ##########################################################################################################################
        ####### compute quantity of virus from the infected birds (in sites only, not farms)
        dico_nodeid_virusinenvBIRDS = {}
        li_sites = [nodeid for nodeid in self.G.nodes() if 'site' in self.G.nodes[nodeid]['nodetype']]
        for nodeid in li_sites:

            ####### update the info on the birds resting in the nodes
            self.dico_step_nodeid_birdidINFO[self.step_id][nodeid] = {birdid.unique_id: {'shedding_rate':birdid.shedding_rate, 
            'state':birdid.state, 'scalingsheddingrateofDeadbirds':np.exp(-self.deadbirdscalingshedding*\
            self.count_consecutive_x_at_end(li=birdid.li_states,x='D')), 'proptimerestingtoday':birdid.proptimerestingtoday,
            'proptimeexploring1node':birdid.proptimeexploring1node} for birdid in self.grid.get_cell_list_contents([nodeid])\
                                                                      if isinstance(birdid, BirdsAgent) and \
                                                                      birdid.remaining_distance2fly==0} 
           
            ####### we assume initially no virus in the environment else compute recursively
            if self.step_id==-1:
                #model starts with a day where no birds moved and are infected as defined by the initial conditions
                dico_nodeid_virusinenvBIRDS[nodeid] = 0
            else:                                   
                ###### From prior step
                V_prior = self.dico_step_nodeid_virusinenv[self.step_id-1][nodeid]
                
                ###### From current infected birds ALIVE
                birdidINFO = self.dico_step_nodeid_birdidINFO[self.step_id][nodeid]
                newvirus_shedding_restinginfected = sum(dico_info_value['shedding_rate'] * \
                                                 (1 if dico_info_value['state'] == 'INFECTED' else 0) *\
                                                 dico_info_value['proptimerestingtoday'] for birdid, dico_info_value in\
                                                 birdidINFO.items())

                ###### From current infected DEAD birds (note: all dead birds died due to infection)
                newvirus_shedding_dead = sum(dico_info_value['shedding_rate'] * \
                                             (1 if dico_info_value['state'] == 'DEAD' else 0) *\
                                             dico_info_value['scalingsheddingrateofDeadbirds'] for birdid, dico_info_value in \
                                             birdidINFO.items())
                
                ###### From current exploring birds (if any)
                newvirus_shedding_exploringinfected = 0
                if nodeid in self.dico_nodeid_liEinfectedbird:
                    newvirus_shedding_exploringinfected = sum(self.dico_birdid_info[agent]['proptimeexploring1node']*\
                                                              self.dico_birdid_info[agent]['shedding_rate'] for agent in\
                                                              self.dico_nodeid_liEinfectedbird[nodeid]) 
                dico_nodeid_virusinenvBIRDS[nodeid] = (1-self.G.nodes[nodeid]['decayrate'])*\
                                                        (V_prior+\
                                                         newvirus_shedding_restinginfected+\
                                                         newvirus_shedding_dead+\
                                                         newvirus_shedding_exploringinfected)           
                

        ##########################################################################################################################
        ################# Add & remove viruses from waterbodies flowing towards s, and those flowing away from s #################
        ##########################################################################################################################
        # Find all the sites towards nodeid, then for each of the connecting edges get the 'ck' attribute and multiply these ck by:
        #self.dico_step_nodeid_virusinenv[self.step_id][nodeid]     
        # By doing this after the impact of infected birds in sites, we assume a one day delay between the virus coming from 
        #waterbodies into sites until these can moves away from resting sites.
        for nodeid in li_sites:         
            
            # Compute new virus from infected sites flowing into the node from connected water bodies
            # edge[0] is the source node, not nodeid
            newvirus_flowingINwaterbodies =sum(self.G.edges[edge]['ck']*((1-self.G.nodes[edge[0]]['decayrate'])**self.G.edges[edge]['T'])*\
                                               self.dico_step_nodeid_virusinenv[self.step_id - self.G.edges[edge]['T']][edge[0]] for\
                                               edge in self.G.in_edges(nodeid) if ('site' in self.G.nodes[edge[0]]['nodetype']) and\
                                               (self.G.edges[edge]['waterspeed']>0) and (self.step_id >= self.G.edges[edge]['T']))
            
            # Compute virus removed from the current node due to water bodies flowing away
            prop_removedvirus = sum([self.G.edges[edge]['ck'] for edge in self.G.out_edges(nodeid) if (self.G.edges[edge]['waterspeed']>0)])

            # Update quantity of virus in the current node
            self.dico_step_nodeid_virusinenv[self.step_id][nodeid] = (1-prop_removedvirus)*(dico_nodeid_virusinenvBIRDS[nodeid]) +\
                                                                     newvirus_flowingINwaterbodies
            
    
    def step(self):

        #print('-------------------------------------')
        #print(self.step_id)
        # trigger each agent to execute its actions (disease dynamics, then behaviour)
        self.schedule.step()

        #at the end of the time step save info for the virus in environment and bird population
        self.saveinfo4diseasedynamic() #should come prior the update_virus_in_environment, as we will use it in update_virus_in_environment
        self.update_virus_in_environment()   
        
        # collect all the data for the given model object
        self.datacollector.collect(self)  
        
        # stop the model when there is less than 5% of alive birds not in settlement sites
        # Count alive birds & alive not in settelemnt birds
        li_alive_birds = [a for a in self.agents if isinstance(a, BirdsAgent) and a.state!='DEAD']
        li_alive_notsettled_birds = [a for a in self.agents if isinstance(a, BirdsAgent) and a.state!='DEAD'and \
                                     self.G.nodes[a.pos].get('nodetype')!='settlementsite']
        if len(li_alive_notsettled_birds)<len(li_alive_birds)*0.05:
            self.running = False
            
        # else update the stepid for the model & agents
        self.step_id += 1
        for agent in self.schedule.agents:
            agent.step_id = self.step_id



