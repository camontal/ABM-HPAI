import os



#### Choose initial folder where to save anything
path_save_INITfolder = r'C:\Users\camil\OneDrive\Desktop\G1\outputAnalysis'

#### Unique naming of the run, so that all associated saved output gets into a specific folder. typically using after changing the code.
#runname = 'RandomSampling26April' #DEBUG
#runname = 'RandomSamplingMaiV9' #ANALYSIS
#runname = 'PathwaysV2' #ANALYSIS
runname = 'RandomSamplingHPAIprimed' #ANALYSIS

path_save_data = os.path.join(path_save_INITfolder, runname)
os.makedirs(path_save_data, exist_ok=True)  # Creates directory only if it doesn't exist

#### Choose network folder
#path_save_networkspace = os.path.join(path_save_INITfolder, 'NetworkSpace', 'NetworkSpace4debug2') #DEBUG
#path_save_networkspace = os.path.join(path_save_INITfolder, 'NetworkSpace', 'NetworkSpaceMai') #ANALYSIS
path_save_networkspace = os.path.join(path_save_INITfolder, 'NetworkSpace', 'NetworkSpaceJuly') #ANALYSIS



###############################################################
####### define features and responses colors and names ########
###############################################################

######## features
dico_var_color = {
    #'base_transmission_rate': 'firebrick',
    'decayrate':'firebrick',
    'days_resistant':'firebrick',
    'days_exposed': 'firebrick',
    'days_incubation': 'firebrick',
    'days_symptom': 'firebrick',
    'days_infected': 'firebrick',
    'beta_bb_mean': 'firebrick',
    'beta_ebf_mean': 'firebrick',
    'beta_flbf_mean': 'firebrick',
    'beta_eb': 'firebrick',
    'beta_fb': 'firebrick',
    'beta_vf': 'firebrick',
    'beta_ff': 'firebrick',
    'shedding_rate': 'firebrick',
    'deadbirdscalingshedding': 'firebrick',
    'p_it': 'firebrick',
    'excess_TR_naive': 'firebrick',
    'excess_CO_naive': 'firebrick',
    'excess_SH_naive': 'firebrick',
    'proba_dying': 'lightcoral',
    'proba_dying_scalenaive': 'lightcoral',
    'delayedmigrationinfectedLOC': 'lightcoral',
    'delayedmigrationinfectedLOC_scalenaive': 'lightcoral',
    'omega': 'lightcoral',
    'omega_scalenaive': 'lightcoral',
    'generalproptimeresting_I': 'lightcoral',
    'generalproptimeresting_I_scalenaive': 'lightcoral',
    'home_range_area_I': 'lightcoral',
    'home_range_area_I_scalenaive': 'lightcoral',
    'SYM': 'lightcoral',
    'watertemperature': 'blue',
    'pH': 'blue',
    'nbr_waterS2F':'blue',
    'nbr_waterS2S':'blue',
    'ck': 'blue', 
    'spatial_synchrony_jaccard_mean': 'darkgoldenrod',
    'sum_acrossstep_nbr_interaction_div1000': 'darkgoldenrod',
    'expected_MDOS': 'darkkhaki',
    'generalproptimeresting_mean': 'darkkhaki',
    'beta_bird_sd': 'darkkhaki',
    'prop_naive': 'khaki',
    'initial_immune_proportion': 'khaki',
    'xi_probadetectfarm': 'silver',
    'sanitation': 'silver'}

#also used for order
dico_col_name = {'darkkhaki':'Individual behaviour',
                 'darkgoldenrod':'Group-level migratory cohesion',
                 'khaki':'Population immunity',
                 'firebrick':'Epidemiological characteristics',
                'lightcoral':'Effect of AI on waterfowl',
                 'blue':'Water properties',
                 'silver':'Biosecurity measures',
                 'deepskyblue':'Landscape'}

dico_col_gr = {'firebrick':'virus',
                'lightcoral':'virus',
                 'blue':'environment',
                 'deepskyblue':'environment',
                 'darkgoldenrod':'waterfowl',
                 'darkkhaki':'waterfowl',
                 'khaki':'waterfowl',
                 'silver':'poultry'}

dico_var_name = {#'base_transmission_rate': 'Base transmission rate',
                 'decayrate':'decay rate',
                'days_resistant':'Post-infection immunity duration',
                'days_exposed': 'Latent period',
                'days_incubation': 'Incubation period',
                'days_symptom': 'Clinical signs period',
                'days_infected': 'Infectious period',
                'beta_bb_mean': 'Transmission rate (bird-to-bird)',
                'beta_ebf_mean': 'Transmission rate (exploratory bird-to-farm)',
                'beta_flbf_mean': 'Transmission rate (flying bird-to-farm)',
                'beta_eb': 'Transmission rate (environment-to-birds)',
                'beta_fb': 'Transmission rate (farms-to-birds)',
                'beta_vf': 'Transmission rate (environment-to-farm)',
                'beta_ff': 'Transmission rate (farm-to-farm)',
                'shedding_rate': 'Shedding rate',
                'deadbirdscalingshedding': 'Duration of infectiousness of dead birds',
                'p_it': 'Residual infection risk',
                'proba_dying': 'Mortality rates ',
                'proba_dying_scalenaive': 'Mortality rates (naive)',
                'delayedmigrationinfectedLOC': 'Delayed migration in infected birds',
                'delayedmigrationinfectedLOC_scalenaive': 'Delayed migration (naive)',
                'omega': 'Prolonged stopover in infected birds',
                'omega_scalenaive': 'Prolonged stopover (naive)',
                'generalproptimeresting_I': 'Reduced movements around stopover sites in infected birds',
                'generalproptimeresting_I_scalenaive': 'Reduced movements around stopover sites (naive)',
                'home_range_area_I': 'Use of smaller areas around stopover sites in infected birds',
                'home_range_area_I_scalenaive': 'Use of smaller areas around stopover sites (naive)',
                'SYM': 'Proportion of birds exhibiting clinical signs',
                'excess_TR_naive': 'Excess contraction by immunologically naïve birds ',
                'excess_CO_naive': 'Excess shedding by immunologically naïve birds ',
                'excess_SH_naive': 'Excess transmission by immunologically naïve birds',
                'watertemperature': 'Water temperature',
                'pH': 'pH',
                'nbr_waterS2F':'Water connectivity stopovers - surrounding farm environment',
                'nbr_waterS2S':'Water connectivity between stopovers',
                'ck': 'Strength of water connectivity', 
                'spatial_synchrony_jaccard_mean': 'Spatial alignment ',
                'sum_acrossstep_nbr_interaction_div1000': 'Bird aggregation',
                'expected_MDOS': 'Stopover duration',
                'generalproptimeresting_mean': 'Prop. time spent resting at stopover vs ‘exploring’ surrounding areas',
                'beta_bird_sd': 'Individual heterogeneity in infectiousness',
                'prop_naive': 'Proportion of immunologically naïve birds at the start of migration',
                'initial_immune_proportion': 'Proportion of immune birds during migration at the start of migration',
                'xi_probadetectfarm': 'Surveillance level',
                'sanitation': 'Sanitation level'}
li_ordered_vars = ['expected_MDOS', 'generalproptimeresting_mean', 'beta_bird_sd', 
                   'spatial_synchrony_jaccard_mean', 'sum_acrossstep_nbr_interaction_div1000', 
                   'prop_naive', 'initial_immune_proportion', 
                   'decayrate', 'days_resistant', 'days_exposed', 'days_incubation', 'days_symptom', 'days_infected', 'beta_bb_mean',
                   'beta_ebf_mean', 'beta_flbf_mean', 'beta_eb', 'beta_fb', 'beta_vf', 'beta_ff', 'shedding_rate', 'deadbirdscalingshedding', 'p_it',
                   'excess_TR_naive', 'excess_CO_naive', 'excess_SH_naive',
                   'proba_dying', 'proba_dying_scalenaive', 'delayedmigrationinfectedLOC', 'delayedmigrationinfectedLOC_scalenaive', 
                   'omega', 'omega_scalenaive', 'generalproptimeresting_I', 'generalproptimeresting_I_scalenaive', 
                   'home_range_area_I', 'home_range_area_I_scalenaive', 'SYM', 
                   'watertemperature', 'pH', 'nbr_waterS2F', 'nbr_waterS2S', 'ck',
                   'xi_probadetectfarm', 'sanitation']

######## response ########
#### blue - waterfowl
#3D3DCC – noticeably darker, rich indigo tone
#5050E0 – still dark but more vibrant blue-purple
#5F5FFF – close to #7171FF but a touch deeper
#8A8AFF – slightly lighter, more pastel version
#### orange: landscape:
#663300 – darkest
#9C5400 – dark
#E67E00 – light
#FFD18C – lightest
dico_response_color = {'AIVmaintenance': '#2E2E99',
                        'max_percinfectedbirds_atonce': '#7A7AFF',
                        'perc_birds_infected': '#D8D8FF',
                        'mean_step_ofthepeak':'#0000FF',
                        'perc_farms_infected': '#C8C800',
                        'spatial_spread_withRe_bigger1':'#9C5400',
                        'duration_withRe_bigger1':'#E67E00',
                        'nbrnodes_Re_bigger1':'#4D2600',
                        'median_y_with_re_bigger1':'#FFD18C'}

dico_response_name = {'perc_farms_infected':'Incursion risk into poultry',#'Farm infection prevalence during migration',
                      'perc_birds_infected':'Prevalence in waterfowl',#'Waterfowl infection prevalence during migration',
                      'max_percinfectedbirds_atonce':'Peak magnitude',
                      'mean_step_ofthepeak':'Peak timing',
                      'AIVmaintenance':'Infection maintenance in waterfowl',
                      'nbrnodes_Re_bigger1':'Site-level prevalence', #'Number of sites with Re>1',
                      'duration_withRe_bigger1':'Infection maintenance in environment', #'Number of days with at least one site with Re>1',
                      'spatial_spread_withRe_bigger1':'Spatial spread',#'Spatial spread of sites with Re>1',
                      'median_y_with_re_bigger1':'Latitudinal positioning'}#'Latitudinal position of sites with Re>1'} 

li_orderedmodels = ['AIVmaintenance', 'mean_step_ofthepeak', 'max_percinfectedbirds_atonce', 'perc_birds_infected',
                    'perc_farms_infected', 
                    'nbrnodes_Re_bigger1','spatial_spread_withRe_bigger1','duration_withRe_bigger1','median_y_with_re_bigger1']








