#%%

from enum import unique
import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
import itertools

class Hypergraph(xgi.Hypergraph):
    def add_naming_game_node(self, label, vocab, committed=False, beta=1, meta=None):
        """Adds a specified selection of nodes to the hypergraph.
        The vocabularies of the nodes can be specified.

        Args:
            uncom_recipe (dict): Specification of uncommitted nodes to be added. Format={<number to be added, type=int>:<vocabulary, type=list}
            com_recipe (dict): Specification of committed nodes to be added. Format={<number to be added, type=int>:<vocabulary, type=list}

        Returns:
            None
        """
        # label = self.num_nodes
        self.add_nodes_from([(label, {'vocab':vocab, 'committed':committed, 'beta':beta, 'metadata':meta})])
        return None
    
    def clear_edges(self):
        """Remove all edges from the graph without altering any nodes."""
        for node in self.nodes:
            self._node[node] = set()
        self._edge.clear()
        self._edge_attr.clear()

    def interact_and_advance(self, edges, frame_num, rule='Unanimous', verbose=False):
        """Function which carries out one interaction in the higher-order naming game on a hypergraph and advances to the next frame.

        Args:
            edges (list): The full, time-resolved list of edges for the hypergraph. Shape should be (number of timesteps, number of edges in each timestep, number of nodes in each edge), expected to be ragged list of lists.
            frame_num (int): The index that determines which timestep is being considered.
            rule (str, optional): A description of the rule used to determine whether local consensus is possible in a given interaction. Defaults to 'Unanimous'.
            show (bool, optional): Whether or not the specifics of each interaction is shown. Useful for debugging. Defaults to False.
        
        Returns:
            None
        """
        self.clear_edges()
        self.add_edges_from(edges[frame_num])
        edge = list(random.choice(self.edges.members()))
        speaker = random.choice(edge)
        if verbose:
            print(f'{self.edges.members()}')
            print(f'Frame number: {frame_num}')
            print(f'Edge: {edge}')
            print(f'Speaker: {speaker}')
        edge.remove(speaker)
        broadcast = random.choice(self.get_attr(speaker, 'vocab'))

        test_stat = np.random.binomial(1, self.get_attr(speaker, 'beta'))

        if rule == 'Unanimous':
            if all([broadcast in self.get_attr(i, 'vocab') for i in edge]):
                if test_stat:
                    for i in edge:
                        if not self.get_attr(i, 'committed'):
                            xgi.classes.function.set_node_attributes(self, {i:{'vocab':[broadcast]}})
                    if not self.get_attr(speaker, 'committed'):
                        xgi.classes.function.set_node_attributes(self, {speaker:{'vocab':[broadcast]}})
            else:
                for i in edge:
                    if broadcast not in self.get_attr(i, 'vocab') and not self.get_attr(i, 'committed'):
                        self.get_attr(i, 'vocab').append(broadcast)
    
        if verbose:
            print(f'Word broadcast: {broadcast}')
            print(f'State of system after interaction: {list(self.nodes.attrs)}')

    def get_attr(self, node, attr):
        """Function to easily obtain a given attribute from a given node in a hypergraph.

        Args:
            node (int): Integer label of the node being interrogated.
            attr (str): Name of the required attribute.

        Returns:
            (Various types): The requested attribute of the given node.
        """
        return list(xgi.stats.nodestats.attrs(self, [node], attr).values())[0]
    
    def count_by_attr(self, attr, val, norm=False):
        if norm:
            return len(self.nodes.filterby_attr(attr, val))/self.num_nodes
        else:
            return len(self.nodes.filterby_attr(attr, val))

def run_naming_game(H, edges, runlength, verbose=False):
    vocab_props = {'A':[], 'B':[], 'AB':[]}
    vocab_props['A'].append(H.count_by_attr('vocab', ['A'], True))
    vocab_props['B'].append(H.count_by_attr('vocab', ['B'], True))
    vocab_props['AB'].append(H.count_by_attr('vocab', ['A', 'B'], True)+H.count_by_attr('vocab', ['B', 'A'], True))
    for i in range(runlength):
        H.interact_and_advance(edges, 0, verbose=verbose)

        vocab_props['A'].append(H.count_by_attr('vocab', ['A'], True))
        vocab_props['B'].append(H.count_by_attr('vocab', ['B'], True))
        vocab_props['AB'].append(H.count_by_attr('vocab', ['A', 'B'], True)+H.count_by_attr('vocab', ['B', 'A'], True))

    return vocab_props

# edges = [[[15,16], [15,16,17]],
#         [[16,17], [15,16,17]],
#         [[15,17], [15,16,17]]]

import json
with open('data/aggr_15min_cliques_thr2_InVS13.json') as json_file:
   edges = [json.load(json_file)]

#print(edges)
edges_flat_1 = list(itertools.chain(*edges))
edges_flat_2 = list(itertools.chain(*edges_flat_1))
unique_id = list(set(edges_flat_2))
print(unique_id)
print(len(unique_id))


H = Hypergraph()
prop_committed = 0.1

number_committed = int(len(unique_id)*prop_committed)
committed_nodes = []
for i in range(number_committed):
    rand_index = random.randint(0, len(unique_id)-1)
    committed_nodes.append(unique_id.pop(rand_index))


for i in unique_id:
    H.add_naming_game_node(i, ['A'], False, beta=1)
for i in committed_nodes:
    H.add_naming_game_node(i, ['B'], True, beta=0)
stats = run_naming_game(H, edges, 500, False)

plt.figure(1)
plt.plot(stats['A'], label='A')
plt.plot(stats['B'], label='B')
plt.plot(stats['AB'], label='AB')
plt.xlabel('Number of interactions')
plt.ylabel('Proportion of population')
plt.legend()


#%%