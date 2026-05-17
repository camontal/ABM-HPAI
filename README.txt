

----------------------------------------------------------------------------------------------------------------
------------------------------------------------ CODE Structure ------------------------------------------------
----------------------------------------------------------------------------------------------------------------

-------- Notebook and python files

ABM_Agents.py: Defines the (wild) bird and poultry agents, including beahvioural rules (only for bird) and disease dynamics (for both poultry and bird)

ABM_DataCollection.py: Handles data collection at each time step, including agent-level, node-level, and model-level summaries. If debugging is enabled, detailed summaries on the individual level is also recorded for every step (e.g., including the conditions the bird is experiencing at that time).

ABM_Model.py: Main model file calling both ABM_Agents.py and ABM_DataCollection.py. It sets up the agent-based model.

config.py: Contains general paths used across scripts

my_batchrunner_patch.py: Custom BatchRunner (i.e., not using the default one from the mesa package) used to run models based on predifed parameter space 

utils.py: Includes utility functions such as functions to generate networks, visualize them and funciton to process data

ABM-simulations.ipynb: Example of how to runs the simulations calling ABM_Model.py, config.py, my_batchrunner_patch.py, utils.py.


----------------------------------------------------------------------------------------------------------------
------------------------------------------------ DATA structure ------------------------------------------------
----------------------------------------------------------------------------------------------------------------
folders within the config.path_save_INITfolder: 

"Networkspace" --> one folder per run containing many graphs and a csv file summarising the graphs attributes
"config.runname" --> N1000 --> allbatched --> 3456789087654 --> 0_df_res.parquet & 0_dico_step_nodeid_virusinenv.pkl
                                                           --> 1_df_res.parquet & 1_dico_step_nodeid_virusinenv.pkl
                                                           -->...
                                         --> 3456789876549 --> 0_df_res.parquet & 0_dico_step_nodeid_virusinenv.pkl
                                                           --> 1_df_res.parquet & 1_dico_step_nodeid_virusinenv.pkl
                                                           -->...
                                         ...
                              DEBUGGING
                              DEBUGGINGVIDEO
                              Graph
                              Visual
                --> N20    --> allbatched --> 3456789087654 --> 0_df_res.parquet & 0_dico_step_nodeid_virusinenv.pkl
                                                         --> 1_df_res.parquet & 1_dico_step_nodeid_virusinenv.pkl
                                                         --> ...
                                       --> ...
                              DEBUGGING
                              DEBUGGINGVIDEO
                              Graph
                              Visual



#NOTE: the unique_id used by mesa, is actually the unique_id, in our case the uniqueunique_id will need to account
#for runid, bathcid. SO DONT CONFUSE BOTH


----------------------------------------------------------------------------------------------------------------
------------------------------------------------ INFO important ------------------------------------------------
----------------------------------------------------------------------------------------------------------------
proba_startmigration, temporal_synchrony, spatial_synchrony, tradeoff_mean, tradeoff_sd --> 'spatial_synchrony_jaccard_mean','sum_acrossstep_nbr_interaction_div100000'
we use these 3 parameters, but then only look at the effective synchrony, i.e. what actually happened.

same for MDOS

Same for number of individuals with symptoms: param: SYM

'perc_site2farm_connection','perc_site2site_connection' --> nbr_waterS2F, nbr_waterS2S
