import networkx as nx
import numpy as np
import scipy 
import xmltodict
import matplotlib.pyplot as plt

### Directly copied from Root Analyzer

def add_root_path(G_root, root_dict, root_id, split = 'left', root_deg = 0):
    
    """
    helper func called by 'rsml2nx', add primary or lateral root to the graph
    - node get the same name as its position.
    - node attr: {t       :  appiration time, 
                 root_id :  root_id,
                 split   :  'left' or 'right'
                 root_deg:  0 if pirmary else 1}
    - edge attr:{ length  :  length,
                  t       :  the appiration time of the later node,
                  split   :  'left' or 'right'}
    - added the root length to graph attribute, the list of root length.
    """
    # add node:
    if type(root_dict) == list:
        for i, node in enumerate(root_dict):
            coord_t = float(node['@coord_t'])
            if coord_t == 0:
                time = 0.0
            elif coord_t < 1:
                time = 0.0
            else:
                time = (coord_t - 1) * 6
            G_root.add_node((float(node['@coord_x']), float(node['@coord_y'])),
                            t = time,
                            th = float(node['@coord_th']),
                            root_id = root_id,
                            split = split,
                            root_deg = root_deg)
        

        # add edge:
        total_length = 0
        for i in range(len(root_dict)-1):
            u = (float(root_dict[i]['@coord_x']), float(root_dict[i]['@coord_y']))
            v =  (float(root_dict[i+1]['@coord_x']), float(root_dict[i+1]['@coord_y']))
            seg_length = scipy.spatial.distance.euclidean(u, v)
            G_root.add_edge(u, v, length = seg_length, t = max( G_root.nodes[u]['t'], G_root.nodes[v]['t']), split = split)
            total_length += seg_length

        G_root.graph[f'lengths_{split}'].append(total_length)
        
    return

def rsml2nx(file):
    """
    RSML to nx directed graph. Left and right root nodes are labeled with "split" attribute. 
    
    --------
    INPUT:
        file  : string, RSML file path/name in the local directory
        
    --------
    OUTPUT: 
        G_root    : nx graph object for the RSML+t Hirros pipeline output
        RSA_dict  : python dict, the dictionary version of the RSML file.
    """

    # convert xml to dict:
    with open(file, 'r', encoding='utf-8') as file:
        rsml = file.read()

    RSA_dict = xmltodict.parse(rsml)
    
    G_root = nx.DiGraph()
    
    G_root.graph['lengths_left'] = []
    G_root.graph['lengths_right'] = []
    
    
    if  RSA_dict['rsml']['scene']['plant'][0]['root']['geometry']['polyline']['point'][0]['@coord_x'] < RSA_dict['rsml']['scene']['plant'][1]['root']['geometry']['polyline']['point'][0]['@coord_x']:       
        # read left:
        left = RSA_dict['rsml']['scene']['plant'][0]
        # read right:
        right = RSA_dict['rsml']['scene']['plant'][1]
    else:
         # read right:
        right = RSA_dict['rsml']['scene']['plant'][0]
        # read left:
        left = RSA_dict['rsml']['scene']['plant'][1]

    # primary root:
    primary = left['root']['geometry']['polyline']['point']
    add_root_path(G_root, primary, root_id = left['root']['@ID'], split = 'left', root_deg = 0)

    # lateral roots:
    if 'root' in left['root']:
        if isinstance(left['root']['root'], list):
            for lr in left['root']['root']:
                # for each lateral:
                lateral = lr['geometry']['polyline']['point']
                add_root_path(G_root, lateral, root_id = lr['@ID'], split = 'left', root_deg = 1)
        else:
            lr = left['root']['root']
            lateral = lr['geometry']['polyline']['point']
            add_root_path(G_root, lateral, root_id = lr['@ID'], split = 'left', root_deg = 1)

    # primary root:
    primary = right['root']['geometry']['polyline']['point']
    add_root_path(G_root, primary, root_id = right['root']['@ID'], split = 'right', root_deg = 0)
    
    # lateral roots:
    if 'root' in right['root']:
        if isinstance(right['root']['root'], list):
            for lr in right['root']['root']:
                # for each lateral:
                lateral = lr['geometry']['polyline']['point']
                add_root_path(G_root, lateral, root_id = lr['@ID'], split = 'right', root_deg = 1)
        else:
            lr = right['root']['root']
            lateral = lr['geometry']['polyline']['point']
            add_root_path(G_root, lateral, root_id = lr['@ID'], split = 'left', root_deg = 1)
          
    # remove self loops:
    G_root.remove_edges_from(nx.selfloop_edges(G_root))
    pos = np.array(list(G_root.nodes()))

    
    # save plotting ratio:
    mins = np.min(pos, 0)
    maxes = np.max(pos, 0)

    G_root.graph['x_min'] = mins[0]
    G_root.graph['y_min'] = mins[1]

    G_root.graph['x_max'] = maxes[0]
    G_root.graph['y_max'] = maxes[1]

    G_root.graph['ratio'] = (maxes[0] - mins[0]) / (maxes[1] - mins[1])
    
    return G_root, RSA_dict


def plot_rsmlt(G_root, plate, save_name=None):

    """
    plot root graph with node color coded by appriation time
    """
    
    pos = np.array(list(G_root.nodes()))

    node_positions = {}
    node_color = []

    for node in G_root.nodes:
        node_positions[node] = ( node[0], - node[1])
        node_color.append(G_root.nodes[node]['t'])

    fig, ax = plt.subplots(figsize = (14,10/G_root.graph['ratio']))

    nx.draw(G_root, pos=node_positions, node_size = 0, width = .7, arrowsize = 5, ax = ax)
    split_root_node = nx.draw_networkx_nodes(G_root, pos=node_positions, node_size= 10, node_color = node_color, ax = ax,  cmap=plt.cm.magma) 
    ax.set_title(f'RSML + t, plate {plate}', fontsize = 20)
    # plt.colorbar(split_root_node, aspect= 50, shrink = .5, label = 'appiration time (h)', orientation = 'vertical')
    cbar = plt.colorbar(split_root_node, aspect=50, shrink=.5, label='aspiration time (h)', orientation='vertical')
    cbar.set_ticks([])  # This removes the tick marks and their labels
    plt.tight_layout()
    plt.show()  
    
    if save_name is not None:
        fig.savefig(save_name)

def plot_rsmlt0(G_root, plate, save_name=None):

    """
    plot root graph subset with nodes where t == 0 (no color coding needed)
    note that it is not th, and might not relfext the most accurate timing
    """
    
    # Create subgraph with only nodes where t == 0
    # note that it is not th, and might not relfext the most accurate timing
    nodes_t0 = [node for node in G_root.nodes if G_root.nodes[node]['t'] == 0]
    G_subset = G_root.subgraph(nodes_t0)
    
    pos = np.array(list(G_subset.nodes()))
    
    # Calculate plotting ratio for the subgraph
    mins = np.min(pos, 0)
    maxes = np.max(pos, 0)
    
    subset_ratio = (maxes[0] - mins[0]) / (maxes[1] - mins[1])

    node_positions = {}
    node_colors = []

    for node in G_subset.nodes:
        node_positions[node] = (node[0], -node[1])
        # Color nodes based on root_deg
        if G_subset.nodes[node]['root_deg'] == 0:
            node_colors.append('purple')
        elif G_subset.nodes[node]['root_deg'] == 1:
            node_colors.append('crimson')
        # else:
        #     node_colors.append('blue')  # fallback color for other root_deg values

    fig, ax = plt.subplots(figsize=(5, 5/subset_ratio))

    nx.draw(G_subset, pos=node_positions, node_size=10, width=.7, arrowsize=5, ax=ax, node_color=node_colors)
    ax.set_title(f'RSML t=0, plate {plate}', fontsize=20)
    plt.tight_layout()
    plt.show()  
    
    if save_name is not None:
        fig.savefig(save_name)

def get_growth(G_root):
    """
    Get 
    --------
    imput:
        G_root  : nx graph object, split root graph
    --------
    output: 
        left_growth    : list of additional length of the left side at each time interval 
        left_tot       : list of cumulative length of the left side 
        right_growth   : list of additional length of the right side at each time interval 
        right_tot      : list of cumulative length of the right side
        
    """

    # create empty list of list for lengths of root segments at different time
    left_lengths = [ [] for _ in range(187//6) ]

    # add the segments
    for edge in G_root.edges:
        if G_root.edges[edge]['split'] == 'left':
            length = G_root.edges[edge]['length']
            print(G_root.edges[edge]['t'])
            left_lengths[G_root.edges[edge]['t']//6-1].append(length)
    
    # get total additional length at each time interval
    left_growth = []
    for i in range(len(left_lengths)):
        tot = np.array(left_lengths[i]).sum()
        left_growth.append(tot)
    
    # get the running total
    left_tot = []
    running = 0
    for i in range(len(left_growth)):
        running += left_growth[i]
        left_tot.append(running)


    right_lengths = [ [] for _ in range(187//6) ]

    for edge in G_root.edges:
        if G_root.edges[edge]['split'] == 'right':
            length = G_root.edges[edge]['length']
            right_lengths[G_root.edges[edge]['t']//6-1].append(length)

    right_growth = []
    for i in range(len(right_lengths)):
        tot = np.array(right_lengths[i]).sum()
        right_growth.append(tot)

    right_tot = []
    running = 0
    for i in range(len(right_growth)):
        running += right_growth[i]
        right_tot.append(running)

    return left_lengths, left_tot, right_lengths, right_tot


def plot_growth(left_tot, right_tot, plate, save_name=None):
    """plot cumulative growths over time"""
    
    fig, ax = plt.subplots(figsize = [8,6])
    plt.plot(np.array(left_tot)*79/1000, label = 'left')
    plt.plot(np.array(right_tot)*79/1000, label = 'right')
    plt.legend(fontsize = 14)
    tps = list(range(0, 187, 6))
    xticks = list(range(0, 32, 1))
    ax.set_xticks(xticks[::6])
    ax.set_xticklabels(tps[::6])
    ax.set_xlabel('hours', fontsize = 14)
    ax.set_ylabel('total length (mm)', fontsize = 14)
    ax.set_title(f'{plate} growth curve', fontsize = 16)
    ax.set_ylim([0, 600])
    plt.show()
    
    if save_name is not None:
        fig.savefig(save_name)