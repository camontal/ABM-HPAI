import mesa #ABM


def followedagent_info(model):
    """Collect information on followed birds and their respective nodes"""
    dico_followedbirds_info = {}
    for agent in model.birdid2follow:
        # Get all attributes of the agent as a dictionary (except 'model' to avoid computational load)
        agent_info = vars(agent).copy()
        del agent_info['model']
        # Retrieve current node information
        current_node = agent.pos
        agent_info['nodeid'] = current_node
        
        #add info from the graph (e.g., longitude etc)
        agent_info.update(model.G.nodes[current_node])
        
        #to save memory dont store all agentsinfo
        if 'agent' in agent_info:
            del agent_info['agent']
        
        # Store the agent info in the dictionary by agent unique_id
        dico_followedbirds_info[agent.unique_id] = agent_info

    return dico_followedbirds_info


def model_info(model):
    dico_info = {}
    
    #add environmental virus in the node
    dico_info['dico_step_nodeid_virusinenv'] = model.dico_step_nodeid_virusinenv
    if model.debuging:
        dico_info['dico_step_nodeid_birdidINFO'] = model.dico_step_nodeid_birdidINFO  
    
    return dico_info



