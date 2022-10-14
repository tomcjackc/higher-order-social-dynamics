#%%
import xgi
import matplotlib.pyplot as plt
import random

def clear_edges(H):
        """Remove all edges from the graph without altering any nodes."""
        for node in H.nodes:
            H._node[node] = set()
        H._edge.clear()
        H._edge_attr.clear()

def add_normal_nodes(hypergraph, recipe):
    """Adds a specified selection of non-committed nodes to the specified hypergraph. 
    The vocabularies of the nodes can be specified.

    Args:
        hypergraph (xgi.classes.hypergraph.Hypergraph): The hypergraph to which nodes should be added.
        recipe (dict): Specification of nodes to be added. Format={<number to be added, type=int>:<vocabulary, type=list}

    Returns:
        None
    """
    counter = hypergraph.num_nodes
    for num in recipe:
        for i in range(num):
            H.add_nodes_from([(counter, {'vocab':recipe[num], 'committed':False})])
            counter += 1
    return None

def add_committed_nodes(hypergraph, recipe):
    """Adds a specified selection of non-committed nodes to the specified hypergraph. 
    The vocabularies of the nodes can be specified.

    Args:
        hypergraph (xgi.classes.hypergraph.Hypergraph): The hypergraph to which nodes should be added.
        recipe (dict): Specification of nodes to be added. Format={<number to be added, type=int>:<vocabulary, type=list}

    Returns:
        None
    """
    counter = hypergraph.num_nodes
    for num in recipe:
        for i in range(num):
            H.add_nodes_from([(counter, {'vocab':recipe[num], 'committed':True})])
            counter += 1
    return None

def interact_and_advance(H, edges, frame_num, rule='Unanimous', show=False):
    """Function which carries out one interaction in the higher-order naming game on a hypergraph.

    Args:
        H (xgi.classes.hypergraph.Hypergraph): The hypergraph upon which the interaction occurs.
        edges (list): The full, time-resolved list of edges for the hypergraph. Shape should be (number of timesteps, number of edges in each timestep, number of nodes in each edge), expected to be ragged list of lists.
        frame_num (int): The index that determines which timestep is being considered.
        rule (str, optional): A description of the rule used to determine whether local consensus is possible in a given interaction. Defaults to 'Unanimous'.
        show (bool, optional): Whether or not the specifics of each interaction is shown. Useful for debugging. Defaults to False.
    
    Returns:
        None
    """
    clear_edges(H)
    H.add_edges_from(edges[frame_num])
    edge = list(random.choice(H.edges.members()))
    speaker = random.choice(edge)
    if show:
        print()
        print(f'Frame number: {frame_num}')
        print(f'Edge: {edge}')
        print(f'Speaker: {speaker}')
    edge.remove(speaker)
    broadcast = random.choice(get_attr(H, speaker, 'vocab'))

    if rule == 'Unanimous':
        if all([broadcast in get_attr(H, i, 'vocab') for i in edge]):
            for i in edge:
                if not get_attr(H, i, 'committed'):
                    xgi.classes.function.set_node_attributes(H, {i:{'vocab':[broadcast]}})
            if not get_attr(H, speaker, 'committed'):
                xgi.classes.function.set_node_attributes(H, {speaker:{'vocab':[broadcast]}})
        else:
            for i in edge:
                if broadcast not in get_attr(H, i, 'vocab') and not get_attr(H, i, 'committed'):
                    get_attr(H, i, 'vocab').append(broadcast)
    
    if show:
        print(f'Word broadcast: {broadcast}')
        print(f'State of system after interaction: {list(H.nodes.attrs)}')

def get_attr(H, node, attr):
    """Function to easily obtain a given attribute from a given node in a hypergraph.

    Args:
        H (xgi.classes.hypergraph.Hypergraph): Hypergraph that contains the interrogated node.
        node (int): Integer label of the node being interrogated.
        attr (str): Name of the required attribute.

    Returns:
        (Various types): The requested attribute of the given node.
    """
    return list(xgi.stats.nodestats.attrs(H, [node], attr).values())[0]


edges = [[[0,1], [0,1,2]],
        [[1,2], [0,1,2]],
        [[0,2]]]

H = xgi.generators.classic.empty_hypergraph()
add_normal_nodes(H, {2:['A']})
add_committed_nodes(H, {1:['B']})
print(list(H.nodes.attrs))
print()
for i in range(3):
    interact_and_advance(H, edges, i, show=True)








deg_list = H.nodes.degree.aslist()

plt.figure(4)
plt.hist(deg_list)


#%%