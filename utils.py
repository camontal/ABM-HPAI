import numpy as np
import pandas as pd
import os
from collections import Counter
import random
import sys
from math import radians, degrees, sin, cos, sqrt, atan2
import copy #deep copy graph

#import mesa #ABM

import math
from enum import Enum
import networkx as nx #network

#visuals
#from mesa.visualization.modules import NetworkModule, NetworkVisualization
#from mesa.visualization.ModularVisualization import ModularServer

import seaborn as sns
import plotly
#import bokeh

from itertools import groupby #data processing

import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.animation import FFMpegWriter #movies
from matplotlib import cm #colormap
import matplotlib.patches as mpatches



#---------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------- DATA PROCESSING --------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

def count_sequences_of_x(li, x):
    #groupby: "group consecutive elements of the same value in an iterable object, such as a list"
    return sum(1 for key, group in groupby(li) if key == x)


def li_liconsnum(li_num):
    if len(li_num)==0:
        return [], []

    # Group the numbers into consecutive li_linum
    li_num = sorted(li_num)
    li_linum = []
    current_group = [li_num[0]]
    
    for i in range(1, len(li_num)):
        if li_num[i] == li_num[i - 1] + 1:
            current_group.append(li_num[i])
        else:
            li_linum.append(current_group)
            current_group = [li_num[i]]
    # Append the last group
    li_linum.append(current_group)    

    return li_linum
#small example
#li_liconsnum([2, 3, 5, 6, 8, 9]) #[[2, 3], [5, 6], [8, 9]]


def find_consecutive_li_linum(li_num):
    li_linum = li_liconsnum(li_num)
    # Return the number of groups and the mean of each group
    return len(li_linum), [np.mean(group) for group in li_linum]
# Example usage
#numbers = [1, 2, 3, 6, 7, 8, 10, 11, 12, 14, 15]
#group_count, group_means = find_consecutive_li_linum(numbers)#4 , [np.float64(2.0), np.float64(7.0), np.float64(11.0), np.float64(14.5)]
#print(group_count, group_means) #4 [2.0, 7.0, 11.0, 14.5]


#small example: dico = dico_step_nbrinfectedbirds = {0: 0, 1: 0, 2: 3, 3: 4, 4: 0, 5: 5, 6: 6, 7: 0, 8: 7, 9: 8, 10: 0}
def dico_startstep_duration(dico):
    
    #lets get a list of all steps which has a value >0
    li_stepwithmore0 = []
    li_val = [dico[step] for step in range(0,max(dico.keys()))]
    for i,v in enumerate(li_val):
        if v>0:
            li_stepwithmore0.append(i)
    #print(li_stepwithmore0)      # [2, 3, 5, 6, 8, 9]
    
    #lets split this list into list of list, each corresponding to a period (consecutive steps)
    li_listep = li_liconsnum(li_stepwithmore0) #consecutive steps with infection
    #print(li_listep) # [[2, 3], [5, 6], [8, 9]]
    
    #lets get the first indices of all sublist which have max number of  steps
    m = max([len(x) for x in li_listep])
    step = [x[0] for x in li_listep if len(x)==m]
    #print(step) # [2, 5, 8]
    
    return step, m 
#dico_startstep_duration(dico)


#---------------------------------------------------------------------------------------------------------------------
#---------------------------------------------- generating network space ---------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

def areasize(x, y, G, scale_smallest=1):
    # Bounding box for the site (we need for the farms especially, to check if farms are below migration route)
    #we use max_explorativewithoutrest to give some perspective, but basically we jsut dont want to large farms
    size_factor = random.uniform(G.graph['max_explorativewithoutrest']/4*0.05*scale_smallest, 
                                 G.graph['max_explorativewithoutrest']/4*scale_smallest)
    #we dont need an exact specific bound, could be sliglty outside the initial ones
    min_x_node = x - size_factor
    max_x_node = x + size_factor
    min_y_node = y - size_factor
    max_y_node = y + size_factor

    # Compute area
    area_size = (max_x_node - min_x_node) * (max_y_node - min_y_node) 
    #the two x and y distance are in 1000km (10**3), i.e. 1000km, to save memory we used TL=6 10^3km instead of 6000km
    #area size is thus in 1'000'000 (i.e. 10**6) km³, i.e. multiply by 1'000'000 to get in km2
    return area_size, min_x_node, max_x_node, min_y_node, max_y_node


def addorientation(G):
    
    #################### define orientation score
    #computed as the distance of the shortest path to the closest settlement sites that starts from the source and target nodes 
    #from which we subtracted the distance of the shortest path from the source node. We then rescaled these values to a 0-1 
    #range and defined the orientation score as the difference from one, so that higher scores indicate closer proximity to 
    #the settlement grounds. 
    
    #list of settlement sites
    li_settlementsites = [node for node, data in G.nodes(data=True) if data.get('nodetype') == 'settlementsite']
    
    #save for each edge its shortest path distance to any settlement site
    dico_edge_minSP = {}
    
    #to avoid recomputing several time the same shortest path, could save time for large graphs
    dico_n2settlement_SP = {} 
    
    #save all the shortest path from all of its n1-n2 edge to any settlement site, to then subtract the distance of the 
    #shortest path from the source node
    dico_n1_SP = {} 
    
    ####### iterate over every edge, and take the min path to any settlement site
    #we will set orientation = 0 for edges below min_displacementwithoutrest, since birds will be allowed to select edges
    #only when their min_displacementwithoutrest>min_displacementwithoutrest
    li_edge = [[n1,n2] for n1,n2 in G.edges() if \
               (G.edges[(n1,n2)]['waterspeed']==0) and \
               (G.edges[(n1,n2)]['min_distance2patch']>=G.graph['min_displacementwithoutrest']) and\
               (G.nodes[n1]['nodetype'] in ['settlementsite', 'stopoversite', 'startingsite']) and \
               (G.nodes[n2]['nodetype'] in ['settlementsite', 'stopoversite', 'startingsite'])]
    for n1,n2 in li_edge:
        if n1 not in dico_n1_SP:
            dico_n1_SP[n1] = []
        
        #### compute the shortest path to each settlement site, store them in li_SP
        li_SP = []
        #for each of settlement site extract the shortest path from n2 to settlement site and add the n1-n2 distance
        for settlementsite in li_settlementsites:
            # Check if there is a path between the nodes else have it set to nan
            SP = np.nan
            if nx.has_path(G, source=n2, target=settlementsite):
                #dont recompute, if already computed once
                if (n2,settlementsite) not in dico_n2settlement_SP:
                    SP = nx.shortest_path_length(G, source=n2, target=settlementsite, weight='min_distance2patch')
                    #weight: distance,cost (documentation)
                    #add the distance of the actual edge
                    SP = SP + G.get_edge_data(n1,n2)['min_distance2patch'] 
                else:
                    SP = dico_n2settlement_SP[(n2,settlementsite)]
            dico_n2settlement_SP[(n2,settlementsite)] = SP
            li_SP.append(SP)
            
        #### save the distance of the shortest path to any settlementsites, into: dico_edge_minSP and dico_n1_SP
        if len(li_SP)==0:
            print(n1,' ', n2)
            sys.exit()
        v = np.nanmin(li_SP)
        dico_edge_minSP[(n1,n2)] = v
        dico_n1_SP[n1].append(v)

    #subtracted the distance of the shortest path from the source node (otherwise the closest you are from the settlement sites,
    #the more the differences in the gained distance matters)
    dico_edge_minSP_ = {}
    for n1,n2 in li_edge:
        dico_edge_minSP_[(n1,n2)] = dico_edge_minSP[(n1,n2)]-np.nanmin(dico_n1_SP[n1])
    
    #rescaled these values to a 0-1 range and defined the orientation score as the difference from one, so that higher scores 
    #indicate closer proximity to     
    mi = np.nanmin(list(dico_edge_minSP_.values())) 
    if mi!=0:
        print('ERROR check how you compute orientation score')
        sys.exit()
    ma = np.nanmax(list(dico_edge_minSP_.values()))
    #scale from 0-1 and subtract 1 so that each nodes have one best link = 1, and the other varaiying from 0-1 in fct of the gain
    dico_edge_O = {edge:round(1-v/ma,3) if not math.isnan(v) else np.nan for edge,v in dico_edge_minSP_.items()}

    #add it to the graph
    #nx.set_edge_attributes(G, 0.8, "orientation")
    #replace orientation that are np.nan (for farms) by 0 for when we compute the proba of migrating, as this include all nodes (Farm will then have proba 0 since A==0
    dico_edge_O = {edge: (O if not np.isnan(O) else 0) for edge, O in dico_edge_O.items()}
    for n1,n2 in G.edges():
        if [n1,n2] in li_edge:
            #if not an edge to a farm
            if G.nodes[n2]['mean_foodavailability']!=0:
                G.edges[(n1,n2)]["orientation"] = dico_edge_O[(n1,n2)]
            else:
                G.edges[(n1,n2)]["orientation"] = 0 
        else:
            G.edges[(n1,n2)]["orientation"] = 0
            
    return G


def avgshortestpathW2B(G):
    """
    Compute the average shortest path from any starting site to any settlement site,
    considering the edge weight 'min_distance2patch'. Also returns the average number of nodes
    in these shortest paths.
    As the distance is the same in both directions, then starting to settlement or vice versa is the same. so we did one of 
    the 2.
    """
    starting_sites = [n for n, d in G.nodes(data=True) if d['nodetype'] == 'startingsite']
    settlement_sites = [n for n, d in G.nodes(data=True) if d['nodetype'] == 'settlementsite']
    
    path_lengths = []
    node_counts = []
    
    for w in starting_sites:
        for b in settlement_sites:
            try:
                path = nx.shortest_path(G, source=w, target=b, weight='min_distance2patch')
                length = nx.shortest_path_length(G, source=w, target=b, weight='min_distance2patch')
                path_lengths.append(length)
                node_counts.append(len(path))  # Number of nodes in the path
            except nx.NetworkXNoPath:
                continue
    
    avg_path_length = np.mean(path_lengths) if path_lengths else float('inf')
    avg_node_count = np.mean(node_counts) if node_counts else float('inf')
    
    return avg_path_length, avg_node_count


def add_farm_node(G, x, y, min_x, max_x, min_y, max_y, constructiontype):
    """ add one farm node """
    new_node = max(G.nodes()) + 1

    density = random.randint(2, 20)  # /m²
    number = random.randint(100, 20000)   
    area_size, min_x_node, max_x_node, min_y_node, max_y_node = areasize(x=x, y=y, G=G, scale_smallest=0.5)
    
    G.add_node(new_node, x=x, y=y, min_x=min_x_node, max_x=max_x_node, min_y=min_y_node, max_y=max_y_node, 
               area_size=area_size, mean_depth=0, W=0, mean_foodavailability=0, 
               nodetype="duckfarm" if random.random() < 0.5 else "chickenfarm", constructiontype=constructiontype,
               density=density, number=number)

    ###### add links to other site (via explorative beahviour) and other farms (if close neough)
    # Define distance thresholds: connect any 2 farms at distance <=10km 
    #& connected any farm to any site for birds epxlorative beahviour if at distance <=max_explorativewithoutrest
    Maxfarmdistance = 10/(10**3)# 10km

    # Connection rules based on node type
    connection_rules = {"site": G.graph['max_explorativewithoutrest'], "farm": Maxfarmdistance}

    # Connect new farm node to nearby nodes based on type
    for node, data in G.nodes(data=True):
        if node == new_node:
            continue  # Skip self-connection

        for key, max_distance in connection_rules.items():
            if key in data['nodetype']:  # contain 'site' or 'farm'
                dist = math.dist((data['x'], data['y']), (x, y))
                if dist <= max_distance:
                    #if the other node is a site: only link the site to the farm not the other way round else both directions
                    G.add_edge(node, new_node, min_distance2patch=dist, waterspeed=0, ck=0, orientation=0)
                    if key=='farm':
                        G.add_edge(new_node, node, min_distance2patch=dist, waterspeed=0, ck=0, orientation=0)
    return G
                

def AddFarmNodes(G, min_x, max_x, min_y, max_y, mean_farm_connectivity, nbr_farm):
    """Add 20 farm nodes with partial spatial structure guided by mean_farm_connectivity."""
    farm_belowroute = []
    farm_explore = []

    # 1. Add 5 'farmbelowroute' farms, i.e., based on random edges
    for _ in range(int(nbr_farm/4)):
        u, v = random.choice(list(G.edges()))
        w = random.uniform(0.3, 0.7) #avoids placing farms exactly at the node endpoints (keeps them clearly between).
        x = G.nodes[u]['x'] * (1 - w) + G.nodes[v]['x'] * w
        y = G.nodes[u]['y'] * (1 - w) + G.nodes[v]['y'] * w
        G = add_farm_node(G, x, y, min_x, max_x, min_y, max_y, 'initfarmbelowroute')
        farm_belowroute.append(max(G.nodes()) + 1)

    # 2. Add 5 'farm2explore' farms, i.e., near random existing nodes (explorative sites)
    for _ in range(int(nbr_farm/4)):
        base_node = random.choice(list(G.nodes()))
        #lets use polar coordinate system, i.e., specifying a given point in a plane by using a distance and an angle as its 2 coordinates
        #direction (random from 0 to 2π)
        angle = random.uniform(0, 2 * math.pi)
        #distance: radius (random distance from the center, we dont want it too close to the center either)
        distance = random.uniform(G.graph['max_explorativewithoutrest']*0.3, G.graph['max_explorativewithoutrest']*0.99)
        #for the math see: https://en.wikipedia.org/wiki/Polar_coordinate_system
        x = G.nodes[base_node]['x'] + distance * math.cos(angle)
        y = G.nodes[base_node]['y'] + distance * math.sin(angle)
        G = add_farm_node(G, x, y, min_x, max_x, min_y, max_y, 'initfarm2explore')
        farm_explore.append(max(G.nodes()) + 1)

    # 3. Add 5 farms near existing 'farmbelowroute' farms
    for _ in range(int(nbr_farm/4)):
        base_node = random.choice(farm_belowroute)
        angle = random.uniform(0, 2 * math.pi)
        x = G.nodes[base_node]['x'] + mean_farm_connectivity * math.cos(angle)
        y = G.nodes[base_node]['y'] + mean_farm_connectivity * math.sin(angle)
        G = add_farm_node(G, x, y, min_x, max_x, min_y, max_y, 'secondfarmbelowroute')

    # 4. Add 5 farms near existing 'farm2explore' farms
    for _ in range(int(nbr_farm/4)):
        base_node = random.choice(farm_explore)
        angle = random.uniform(0, 2 * math.pi)
        x = G.nodes[base_node]['x'] + mean_farm_connectivity * math.cos(angle)
        y = G.nodes[base_node]['y'] + mean_farm_connectivity * math.sin(angle)
        G = add_farm_node(G, x, y, min_x, max_x, min_y, max_y, 'secondfarm2explore')

    return G

def visual(G, path_, title='', save=True, plot=False, type_='nodetype'):
    '''Visualise the network where the width of the edge is representing one of the two orientation: the larger, 
    the larger scores (actually only when oreitnation =1 we have larger edges). The color of the edges is blue if also has 
    water, else green'''
    
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)}

    # Ensure the directory exists
    os.makedirs(path_, exist_ok=True)
  
    if type_ == 'nodetype':
        dico_type_col = {'stopoversite': 'skyblue', 'settlementsite': 'black', 'startingsite': 'black', 'exploringsite':'green',
                         'chickenfarm': 'orange', 'duckfarm': 'red'}
    elif type_ == 'constructiontype':
        dico_type_col = {'initialsites': 'skyblue', 'exploring': 'green', 
                        'initfarmbelowroute':'yellow', 'secondfarmbelowroute':'gold',
                        'initfarm2explore': 'purple', 'secondfarm2explore':'red'}

    li_x = [data['x'] for _, data in G.nodes(data=True)]
    li_y = [data['y'] for _, data in G.nodes(data=True)]

    # Assign node colors based on type, default to gray
    li_col = [dico_type_col.get(data.get(type_, 'unknown'), 'gray') for _, data in G.nodes(data=True)]

    li_area = np.array([data['area_size'] for _, data in G.nodes(data=True)])

    # Min-max normalization for node sizes
    minS, maxS = 30, 100
    li_area = minS + (li_area - li_area.min()) * (maxS - minS) / (li_area.max() - li_area.min())

    # Set figure width and height proportionally
    fig_width = 4  
    aspect_ratio = (max(li_y) - min(li_y)) / (max(li_x) - min(li_x))
    fig_height = fig_width * aspect_ratio

    # Create figure and axis
    fig, ax = plt.subplots(figsize=(fig_width, fig_height))
    ax.set_xlabel("x")
    ax.set_ylabel("y")
    
    # Plot nodes
    ax.scatter(li_x, li_y, c=li_col, s=li_area, edgecolors='none', alpha=0.8)

    # Custom colormap from black to violet
    cmap = mcolors.LinearSegmentedColormap.from_list("mine", ["lightgrey", "black"])
    norm = mcolors.Normalize(vmin=0, vmax=1)  # Normalize "orientation" between 0 and 1

    # Plot edges with orientation-based color and width
    for u, v, data in G.edges(data=True):
        x1, y1 = pos[u]
        x2, y2 = pos[v]
        orientation = data.get('orientation')  
        edge_color = cmap(norm(orientation))  # Map value to color
        edge_color = 'blue' if data.get('waterspeed')>0 else 'green'
        #edge_width = orientation 
        edge_width = 0.7 if orientation==1 else 0.2
        if edge_color=='blue':
            edge_width = 0.7
        ax.plot([x1, x2], [y1, y2], color=edge_color, alpha=0.8, linestyle='-', lw=edge_width)

    # Add colorbar for edge orientation
    #sm = cm.ScalarMappable(cmap=cmap, norm=norm)
    #sm.set_array([])
    #fig.colorbar(sm, ax=ax, label="Edge Orientation")  # Now explicitly linked to `ax`

    ax.set_title(title)
    ax.grid(True, linestyle='--', alpha=0.5)

    # Create legend for node types
    legend_patches = [mpatches.Patch(color=color, label=nodetype) for nodetype, color in dico_type_col.items()]
    ax.legend(handles=legend_patches, loc='upper left', bbox_to_anchor=(1.02, 1), borderaxespad=0, title="Node Types")
    if save:
        plt.savefig(os.path.join(path_, title + '.png'), bbox_inches='tight', dpi=400)
    if plot:
        plt.show()
    plt.close()

    
def perturb_nodes(G, fixed_nodes, perturbation_scale, TL):
    """
    Randomly moves some nodes (except for fixed ones) within small perturbation bounds.
    Returns a new graph with perturbed coordinates.
    """
    G_new = copy.deepcopy(G)  # Ensure full duplication to prevent side effects

    for node in G_new.nodes:
        if node in fixed_nodes:
            continue  
        dx = random.uniform(-perturbation_scale, perturbation_scale)
        dy = random.uniform(-perturbation_scale, perturbation_scale)
        new_x = max(0, min(G_new.nodes[node]['x'] + dx, TL))
        new_y = max(0, min(G_new.nodes[node]['y'] + dy, TL))
        G_new.nodes[node]['x'] = new_x
        G_new.nodes[node]['y'] = new_y

    return G_new


def add_bridge_nodes(G, nbr_bridgenodes, TL, min_x, max_x, min_y, max_y):
    """
    Add bridge nodes to improve connectivity in the network.
    Node coordinates are adjusted iteratively until the required number of valid pairs is reached.
    """
    nodes_list = list(G.nodes())
    sorted_nodes_by_y = sorted(nodes_list, key=lambda n: G.nodes[n]['y'])
    lowest_y_node = sorted_nodes_by_y[0]
    highest_y_node = sorted_nodes_by_y[-1]
    fixed_nodes = {lowest_y_node, highest_y_node}

    li_pairs = []
    while len(li_pairs) < nbr_bridgenodes:
        G_new = perturb_nodes(G=G, fixed_nodes=fixed_nodes, perturbation_scale=0.01*TL, TL=TL)  # Generate a perturbed version of G
        li_pairsnew = []

        for i, n1 in enumerate(nodes_list):
            for n2 in nodes_list[i+1:]:
                if (G_new.nodes[n1]['nodetype'] == 'stopoversite' and 
                    G_new.nodes[n2]['nodetype'] == 'stopoversite' and not G_new.has_edge(n1, n2)):
                    x1, y1 = G_new.nodes[n1]['x'], G_new.nodes[n1]['y']
                    x2, y2 = G_new.nodes[n2]['x'], G_new.nodes[n2]['y']
                    distance = math.dist((x1, y1), (x2, y2))
                    if distance <= 2 * G.graph['max_explorativewithoutrest']:
                        li_pairsnew.append([n1, n2])

        if len(li_pairsnew) > len(li_pairs):  # Only keep the perturbation if it improves connectivity as desired
            G = G_new
            li_pairs = li_pairsnew

    random.shuffle(li_pairs)
    for n1, n2 in li_pairs[:nbr_bridgenodes]:
        bridge_node = max(G.nodes) + 1
        x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
        x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
        x = (x1 + x2) / 2
        y = (y1 + y2) / 2

        mean_depth = round(random.uniform(0.3, 0.5) / 1000,4)
        mean_foodavailability = round(random.uniform(0.7, 1),2)
        area_size, min_x_node, max_x_node, min_y_node, max_y_node = areasize(x=x, y=y, G=G, scale_smallest=0.5)        
        volume_water = area_size * mean_depth

        G.add_node(bridge_node, x=x, y=y, min_x=min_x_node, max_x=max_x_node,
                   min_y=min_y_node, max_y=max_y_node, area_size=area_size,
                   mean_depth=mean_depth, W=volume_water, mean_foodavailability=mean_foodavailability,
                   nodetype='exploringsite', density=0, number=0, constructiontype='exploring')

        G.add_edge(n1, bridge_node, min_distance2patch=math.dist((x1, y1), (x, y)), waterspeed=0, ck=0, orientation=0)
        G.add_edge(bridge_node, n1, min_distance2patch=math.dist((x1, y1), (x, y)), waterspeed=0, ck=0, orientation=0)
        G.add_edge(bridge_node, n2, min_distance2patch=math.dist((x2, y2), (x, y)), waterspeed=0, ck=0, orientation=0)
        G.add_edge(n2, bridge_node, min_distance2patch=math.dist((x2, y2), (x, y)), waterspeed=0, ck=0, orientation=0)

    return G


def create_random_network(ratiobbox, displflexibility, ratioexplorative, A, nbr_nonbridgenodes, nbr_bridgenodes, TL,
                          mean_farm_connectivity, nbr_farm):
    """
    Create a random directed network where nodes represent sites and edges represent potential bird movement.
    
    Parameters:
        ratiobbox: give the ratio of the rectangle width vs height where we will select node randomly
        displflexibility: small number say between 0.5 and 0 (0  mean two nodes are linked only when at an exact distance of 1/A)
        ratioexplorative: how much of the min distance withoout rest distance should birds be able to fly when just exploring the
        surrounding of stopover sites?
        A: average number of stopover sites
        nbr_nonbridgenodes: int - number of non-bridge nodes
        nbr_bridgenodes: int - Number of bridge nodes allowing interactions between otherwise disconnected nodes.
    
    Returns:
        G: networkx.DiGraph - The generated directed network.
    """
    G = nx.DiGraph()
    
    min_displacementwithoutrest = TL/A * (1 - displflexibility) #birds’ migratory movement range 
    max_displacementwithoutrest = TL/A * (1 + displflexibility) #birds’ migratory movement range 
    max_explorativewithoutrest = min_displacementwithoutrest * ratioexplorative #maximum exploratory distance 
    min_x = 0; max_x = TL*ratiobbox
    min_y = 0; max_y = TL
    li_x = [i/100 for i in list(range(min_x,int(max_x*100)))]
    li_y = [i/100 for i in list(range(min_y,max_y*100))]
        
    #iterate until we have the requested nodes within the largest connected component and keep the first and last node
    scale = 1
    while (len(G.nodes)!=nbr_nonbridgenodes) or (0 not in G.nodes()) or (1 not in G.nodes()):
        random.seed()  # Set a random seed to ensure different results for each run
        if len(G.nodes) < nbr_nonbridgenodes:
            scale = scale * 1.2
        if len(G.nodes) > nbr_nonbridgenodes:
            scale = scale / 1.1
        G = nx.DiGraph()
        G.graph['min_displacementwithoutrest'] = min_displacementwithoutrest
        G.graph['max_displacementwithoutrest'] = max_displacementwithoutrest
        G.graph['max_explorativewithoutrest'] = max_explorativewithoutrest
    
        available_pairs = [(x, y) for y in li_y for x in li_x] 
        
        ############## Add nodes with attributes
        for nodeid in range(int(nbr_nonbridgenodes * scale)):  # Generate extra nodes to ensure full connectivity
            if not available_pairs:
                raise ValueError("Not enough unique (x, y) pairs to satisfy the number of nodes.")
                
            # Select a random (x, y) pair and remove it from available_pairs to avoid duplicates
            x, y = random.choice(available_pairs)
            #for the first two nodes, add them with random x but max and min y
            if nodeid==0:
                y = min(li_y)
            if nodeid==1:
                y = max(li_y)    
            available_pairs.remove((x, y))
                
            #get nodes characteristics
            mean_depth = round(random.uniform(0.0003, 0.002),4)  # 30cm - 2meter, in km
            mean_foodavailability = round(random.uniform(0.7, 1),2)
            pH = round(random.uniform(5, 8.5),1)
            area_size, min_x_node, max_x_node, min_y_node, max_y_node = areasize(x=x, y=y, G=G)
            volume_water = area_size * mean_depth  #in 1'000'000 km³, i.e. multiply by 1'000'000 to get it in km³
            
            G.add_node(nodeid, x=x, y=y, min_x=min_x_node, max_x=max_x_node,
                       min_y=min_y_node, max_y=max_y_node, area_size=area_size,
                       mean_depth=mean_depth, W=volume_water, mean_foodavailability=mean_foodavailability,
                       nodetype="stopoversite", density=0, number=0, constructiontype='initialsites')

        ############## Connect nodes for migratory movements only
        nodes_list = list(G.nodes)
        for i, n1 in enumerate(nodes_list):
            for n2 in nodes_list[i+1:]:
                x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
                x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
                distance = math.dist((x1, y1), (x2, y2))        
                #connect for migratory mvt OR for exploration
                # for migratory movements only: because otherwise there could be some stopover sites which are accessible only through 
                # an edge of distance < max_explorativewithoutrest and then we wont be able to have
                # the migratory movement larger than this in the model, without having birds stuck in some places.
                if (distance >= G.graph['min_displacementwithoutrest'] and distance <= G.graph['max_displacementwithoutrest']):
                    G.add_edge(n1, n2, min_distance2patch=distance, waterspeed=0, ck=0)
                    G.add_edge(n2, n1, min_distance2patch=distance, waterspeed=0, ck=0)

        # Keep the largest connected component and remove unnecessary nodes
        largest_cc = max(nx.weakly_connected_components(G), key=len)
        G = G.subgraph(largest_cc).copy()
        #print('largestcomoponent nbr of nodes: ', len(G.nodes))
    
    #print('scale:',scale)  
    
    ###### flag settlement and starting sites for some of the stopover sites
    # Flag exactly 3 settlement and 3 starting sites based on latitude ('y' attribute)
    # Sort nodes by 'y' coordinate & Get bottom 3 for starting sites, top 3 for settlement sites
    sorted_nodes_by_y = sorted(G.nodes(data=True), key=lambda x: x[1]['y'])#x[0] is the node id, 1, is the dico of attributes
    starting_nodes = sorted_nodes_by_y[:3]
    settlement_nodes = sorted_nodes_by_y[-3:]
    for node, data in G.nodes(data=True):
        if any(node == n[0] for n in starting_nodes):
            data['nodetype'] = 'startingsite'
        elif any(node == n[0] for n in settlement_nodes):
            data['nodetype'] = 'settlementsite'

    ########## add orientaiton score before farms and bridges since birds cant stop on these!
    G = addorientation(G)
            
    ########## Add bridge nodes to connect nodes for explorative birds
    #print('nbr nodes without bridges: ', G.number_of_nodes())
    G = add_bridge_nodes(G=G, nbr_bridgenodes=nbr_bridgenodes, TL=TL, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y)
    
    if G==None:
        #print('could not add as many bridge node, we wont save this network')
        return None
    #print('nbr nodes with bridge nodes: ', G.number_of_nodes())
    
    
    ########## Add exploring edges from any two nodes (either sites or exploring node)
    #to do so, we add edges with same rules as the other stopover sites, so that these bridges nodes can be considered as 
    #stopover sites
    for i, n1 in enumerate(G.nodes):
        for n2 in list(G.nodes)[i+1:]:  # Only nodes after n1 to avoid (2,1) if (1,2) already done
            if n1 == n2:
                continue

            x1, y1 = G.nodes[n1]['x'], G.nodes[n1]['y']
            x2, y2 = G.nodes[n2]['x'], G.nodes[n2]['y']
            distance = math.dist((x1, y1), (x2, y2))

            # Check if it allows for other exploratory or migratory movements
            #(distance >= G.graph['min_displacementwithoutrest'] and distance <= G.graph['max_displacementwithoutrest'])
            if (distance <= G.graph['max_explorativewithoutrest']):

                # Ensure the edge does not already exist before adding
                if not G.has_edge(n1, n2):
                    G.add_edge(n1, n2, min_distance2patch=distance, waterspeed=0, ck=0, orientation=0)
                if not G.has_edge(n2, n1):
                    G.add_edge(n2, n1, min_distance2patch=distance, waterspeed=0, ck=0, orientation=0)

                    
    ########## Scale distances so the longest link is 6,000 km
    #max_distance = max(nx.get_edge_attributes(G, 'min_distance2patch').values())
    #scaling_factor = 6000 / max_distance  # Scale so that longest link = 6,000 km
    
    #for u, v in G.edges():
    #    G.edges[u, v]['min_distance2patch'] *= scaling_factor    
    
    ########## Add farm nodes & water edge (site to farm; site to site, not farm to farm nor farm to site)
    #we added 20 farms, including 10 farms below a link (and connected to sites where eplxorative beahvioru allows) and 10 farms 
    #connected to sites via explorative behaviour (possibly also below a link). We then added a water edge from the farm nodes to
    G = AddFarmNodes(G=G, min_x=min_x, max_x=max_x, min_y=min_y, max_y=max_y, mean_farm_connectivity=mean_farm_connectivity,
                     nbr_farm=nbr_farm)
    return G


def addicephenology(G):
    # Ice end and start ranges
    ice_end_min, ice_end_max = 30, 80
    ice_start_min, ice_start_max = 230, 300

    # Find min and max y-coordinates in the graph
    min_y = min(G.nodes[i]['y'] for i in G.nodes)
    max_y = max(G.nodes[i]['y'] for i in G.nodes)

    # Assign ice_end and ice_start to each node based on y-coordinate
    for i in G.nodes:
        y = G.nodes[i]['y']

        # Interpolate ice_end: 30 for lowest y, 80 for highest y
        ice_end = int(ice_end_min + (y - min_y) * (ice_end_max - ice_end_min) / (max_y - min_y))

        # Interpolate ice_start: 300 for highest y, 230 for lowest y
        ice_start = int(ice_start_max - (y - min_y) * (ice_start_max - ice_start_min) / (max_y - min_y))

        # Assign to the node
        G.nodes[i]['ice_end'] = ice_end
        G.nodes[i]['ice_start'] = ice_start
    return G


#---------------------------------------------------------------------------------------------------------------------
#-------------------------------------------------- NETWORK VISUALS --------------------------------------------------
#---------------------------------------------------------------------------------------------------------------------

def Gplot_mvt(G, df, stepid, ax, DOY, nodesize_scale, edgesize_scale, scale_envnode_size_D, scalingwaterspeededge_D,
              edgeorientation_scale_D, title='', alpha=0.8):
    df = df[(~df['uniqueagentid'].isnull())&(df['state']!='DEAD')]
    if ax==None:
        
        # Set figure width and height proportionally
        fig_width = 4  
        li_x = [data['x'] for _, data in G.nodes(data=True)]
        li_y = [data['y'] for _, data in G.nodes(data=True)]
        aspect_ratio = (max(li_y) - min(li_y)) / (max(li_x) - min(li_x))
        fig_height = fig_width * aspect_ratio
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(fig_width, fig_height))
        
    # Position the nodes 
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)} #according to x and y attributes
    
    ######################## ENVIRONMENT
    #node: have the color of their mean_foodavailability: the more attractive the more green, if 0 then it should be red.
    node_colors = []
    dico_node_colors = {}
    #node_size = []
    dico_node_size = {}
    mean_foodavailability_values = []
    li_mean_foodavailability = []
    for node in G.nodes(data=True):
        area_size = node[1]['area_size']*10**3
        #node_size.append(area_size/8)
        dico_node_size[node[0]] = area_size/scale_envnode_size_D
        mean_foodavailability = node[1]['mean_foodavailability']
        mean_foodavailability_values.append(mean_foodavailability)
        if mean_foodavailability > 0:
            li_mean_foodavailability.append(mean_foodavailability)
            # Scale mean_foodavailability to a range of colors (0 to 1 corresponds to blue shades)
            doy_ = stepid+DOY
            color = plt.cm.YlGn(max(0,mean_foodavailability-0.001))  #not sure why but plt.cm.YlGn(0.99) is very different than plt.cm.YlGn(1)
            #color it blue if still ice!
            if doy_<=node[1]['ice_end']:
                color='blue'
        else:
            color = 'red'  # red for mean_foodavailability = 0
        node_colors.append(color)
        dico_node_colors[node[0]] = color
    # draw nodes with color representing the food availability (cannot have dfferetn shapes of node in a same graph, could plot one graph
    #per nodetype later on though)
    #nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=0.8, ax=ax) #edgecolors=node_colors,
    #nx.draw_networkx_labels(G, pos, font_color='black',font_size=8, ax=ax)
    nx.draw_networkx_labels(G, pos, font_size=dico_node_size, font_color=dico_node_colors, ax=ax)
    
    #draw edge
    #edge: color: should be indicative of their waterflow, so that if >0 then it should be variation of blue, else it should be green
    #edge: size: should be indicative of their orientation: the more orientation the smaller edge. 
    edge_orientation_widths = []
    edge_orientation_colors = []
    edge_waterspeed_widths = []
    edge_waterspeed_colors = []
    for u, v, data in G.edges(data=True):
        orientation = data['orientation']
        waterspeed = data['waterspeed']

        #### waterspeed edges
        if waterspeed > 0:
            # Blue shades based on waterspeed (0 to max)
            edge_waterspeed_colors.append('blue') #plt.cm.Blues(max(0,waterspeed/100-0.001))  # Normalize waterspeed to [0, 1] range
            edge_waterspeed_widths.append(waterspeed/scalingwaterspeededge_D)
        else:
            edge_waterspeed_colors.append('white')
            edge_waterspeed_widths.append(0)

        #### orientation edges
        if orientation>0:
            edge_orientation_colors.append('green')
            edge_orientation_widths.append(orientation/edgeorientation_scale_D) #(2+np.log(orientation))/2)
        else:
            edge_orientation_colors.append('white')
            edge_orientation_widths.append(0)

    # Draw green orientation score edges with width representing how large is the orientation
    # Draw blue waterspeed score edges with width representing how large is the waterspeed
    nx.draw_networkx_edges(G, pos, width=edge_orientation_widths, edge_color=edge_orientation_colors, alpha=0.5, ax=ax,
                           connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edges(G, pos, width=edge_waterspeed_widths, edge_color=edge_waterspeed_colors, alpha=0.5, ax=ax,
                           connectionstyle='arc3,rad=0.1')


    ######################## MOVEMENTS
    # Draw the graph G representing the MOVEMENT from step 1, at step 0 all birds are in the first node and there is no edges
    #remove dead birds
    df_nodes = df[(df['step']==stepid) & (~df['sourcenode'].isnull()) & (~df['targetnode'].isnull()) &\
                 (df['behaviorual_status']!='flying')].copy()
    df_edges = df[(df['step']==stepid) & (df['behaviorual_status']!='resting')].copy()
    
    if stepid!=0:
        
        ########## edges
        #edge width proportional to number of birds that used that edge and color indicative of proportion of infected bird
        li_edges = list(zip(df_edges['sourcenode'], df_edges['targetnode']))
        edge_count = Counter(li_edges)
        unique_edges = list(edge_count.keys())
        mvtedge_widths = [edge_count[edge]*edgesize_scale for edge in unique_edges] 
        nx.draw_networkx_edges(G, pos, edgelist=unique_edges, width=mvtedge_widths,
                               edge_color='orange', alpha=alpha, ax=ax, connectionstyle='arc3,rad=0.1')

        ########## nodes
        # Get nodes and their properties for the current step
        node_counts = df_nodes.groupby(['x','y']).size()
        node_colors_mvt = []
        node_sizes_mvt = []
        minbm = df['m'].min()
        min_node_size = 10
        for (x, y) in pos.values():
            if (x, y) in node_counts: 
                #if only one bird, then track its state as well
                if len(df['uniqueagentid'].unique())==1:
                    body_mass = df_nodes[(df_nodes['y']==y) & (df_nodes['x']==x)]['m'].values[0]
                    state = df_nodes[(df_nodes['y']==y) & (df_nodes['x']==x)]['state'].values[0]
                    if state=='INFECTED':
                        col='red'
                    elif state=='SUSCEPTIBLE':
                        col='yellow'
                    elif state=='EXPOSED':
                        col='turquoise'
                    elif state=='RESISTANT':
                        col='blue'                        
                    elif state=='DEAD':
                        col='red'   
                    else:
                        print('ERROR: CHECK YOUR BIRD STATE')
                        sys.exit()
                    node_colors_mvt.append(col) #state
                    node_sizes_mvt.append(body_mass - minbm + 10) #body_mass
                else:
                    # Use Reds colormap to scale red intensity with proportion infected
                    prop_infected = df_nodes[(df_nodes['y']==y) & \
                                             (df_nodes['x']==x)]['state'].value_counts(normalize=True).get('INFECTED', 0)
                    node_colors_mvt.append(cm.Reds(prop_infected)) #reder, more infections
                    node_sizes_mvt.append(node_counts[(x,y)] * nodesize_scale+min_node_size) #number of birds
            else:
                node_colors_mvt.append('black')
                node_sizes_mvt.append(min_node_size)
        # Draw nodes with networkx
        nx.draw_networkx_nodes(G, pos, node_color=node_colors_mvt, node_size=node_sizes_mvt, edgecolors='black', alpha=alpha, ax=ax)        
        if title!='':
            ax.set_title(title)
    
    ax.grid(True)
    #for spine in ax.spines.values():
    #    spine.set_visible(False)


def Gplot(G, orientationwidthedge_scale=2, site_scale=4):
    ########################## PLOT       
    ######## Generate color map for nodes based on their mean_foodavailability
    #node: have the color of their mean_foodavailability: the more attractive the more green, if 0 then it should be red.
    node_colors = []
    node_size = []
    mean_foodavailability_values = []
    li_mean_foodavailability = []
    for node in G.nodes(data=True):
        area_size = node[1]['area_size']
        node_size.append(area_size/site_scale)
        mean_foodavailability = node[1]['mean_foodavailability']
        mean_foodavailability_values.append(mean_foodavailability)
        if mean_foodavailability > 0:
            li_mean_foodavailability.append(mean_foodavailability)
            # Scale mean_foodavailability to a range of colors (0 to 1 corresponds to blue shades)
            color = plt.cm.YlGn(max(0,mean_foodavailability-0.001))  #not sure why but plt.cm.YlGn(0.99) is very different than plt.cm.YlGn(1)
        else:
            color = '#DCDC00'  # kappa flu farm color for mean_foodavailability = 0
        node_colors.append(color)

    ######## Generate edge attributes for visualization
    #edge: color: should be indicative of their waterflow, so that if >0 then it should be variation of blue, else it should be green
    #edge: size: should be indicative of their orientation: the more orientation the smaller edge. 
    edge_orientation_widths = []
    edge_orientation_colors = []
    edge_waterspeed_widths = []
    edge_waterspeed_colors = []
    for u, v, data in G.edges(data=True):
        orientation = data['orientation']
        waterspeed = data['waterspeed']
        
        #### waterspeed edges
        if waterspeed > 0:
            # Blue shades based on waterspeed (0 to max)
            edge_waterspeed_colors.append('blue') #plt.cm.Blues(max(0,waterspeed/100-0.001))  # Normalize waterspeed to [0, 1] range
            edge_waterspeed_widths.append(waterspeed/50)
        else:
            edge_waterspeed_colors.append('white')
            edge_waterspeed_widths.append(0)
            
        #### orientation edges
        if orientation>0:
            edge_orientation_colors.append('#105C59')
            edge_orientation_widths.append(1+orientation*orientationwidthedge_scale) #(2+np.log(orientation))/2)
        else:
            edge_orientation_colors.append('#105C59')
            edge_orientation_widths.append(1) #still plot since to go to farms, the orientaiton is 0 but birds can access it during foraging


    ######## Create figure and axis
    fig, ax = plt.subplots() #figsize=(5, 8)

    # Position the nodes 
    #pos = nx.spring_layout(G, seed=42) #a spring layout
    pos = {node: (data['x'], data['y']) for node, data in G.nodes(data=True)} #according to x and y attributes

    # Draw nodes with specific color (cannot have dfferetn shapes of node in a same graph, could plot 1 graph per nodetype later on though)
    nx.draw_networkx_nodes(G, pos, node_color=node_colors, node_size=node_size, alpha=0.8, ax=ax)

    # Draw green orientation score edges with width representing how large is the orientation
    # Draw blue waterspeed score edges with width representing how large is the waterspeed
    nx.draw_networkx_edges(G, pos, width=edge_orientation_widths, edge_color=edge_orientation_colors, alpha=0.5, ax=ax, connectionstyle='arc3,rad=0.1')
    nx.draw_networkx_edges(G, pos, width=edge_waterspeed_widths, edge_color=edge_waterspeed_colors, alpha=0.5, ax=ax, connectionstyle='arc3,rad=0.1')

    # Draw nodes ID
    nx.draw_networkx_labels(G, pos, font_size=8, font_color='black', ax=ax)

    # Set axis labels
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.grid(True)