#%%

from enum import unique
import xgi
import matplotlib.pyplot as plt
import random
import numpy as np
import numpy.random as rand
import itertools
from tqdm import tqdm
import csv
import os
import json

def flatten(l):
    return [item for sublist in l for item in sublist]
#%%

# model constructor

class HigherOrderNamingGame(xgi.Hypergraph):
    def __init__(self, rule='Unanimous', incoming_data=None, **attr):
        xgi.Hypergraph.__init__(self, incoming_data, **attr)
        self.rule = rule
        self.rewire_counter = 0
        self.figs = []
    def add_naming_game_node(self, list_nodes, vocab, committed=False, beta=1, q=0, meta=None):
        """Adds a set of identical naming game nodes, with defined vocabularies, levels of
        committment, beta, and optional metadata.

        Args:
            list_nodes (list): list of node labels to be added to the hypergraph.
            vocab (list): list representing the vocabulary shared by the added nodes.
            committed (bool, optional): the committment status of the added nodes. Defaults to False.
            beta (int, optional): the beta value (representing synergy/efficiency of communication/social influence) of the added nodes. Defaults to 1.
            meta (_type_, optional): optional metadata associated with the addded nodes (eg. whether the node is a teacher or a student in a school). Defaults to None.

        Returns:
            None
        """
        # label = self.num_nodes
        N = len(list_nodes)
        
        list_dict = [{'vocab':vocab, 'committed':committed, 'beta':beta, 'q':q}]*N
        self.add_nodes_from(zip(list_nodes, list_dict))
        return None
    
    def clear_edges(self):
        """Remove all edges from the graph without altering any nodes. This is an alteration to the XGI code and was implemented in PR #199."""
        for node in self.nodes:
            self._node[node] = set()
        self._edge.clear()
        self._edge_attr.clear()
    
    def draw(self):
        return None

    #     a = self.nodes.attrs('vocab').aslist()
    #     for i, n in enumerate(a):
    #         if n == ['A']:
    #             a[i] = 'blue'
    #         if n == ['B']:
    #             a[i] = 'orange'
    #         if n == ['A', 'B'] or n == ['B', 'A']:
    #             a[i] = 'white'
    #     print(a)
    #     plt.figure()
    #     print(xgi.drawing.xgi_pylab.draw(self, node_fc = a))
    #     plt.savefig('test.png')
    #     plt.show()
    #     # return fig
            

    def interact_and_advance(self, verbose=False, display=False):
        """Function which carries out one interaction of the naming game on a given edge, and advances to the next frame.

        Args:
            edge (list): list of nodes existing on the selected edge.
            verbose (bool, optional): whether or not the status of the interaction should be displayed. Useful for debugging, impractical for large hypergraphs. Defaults to False.

        Returns:
            dict: dictionary containing the changes made to the counts of each vocabulary, used for counting efficiently.
        """
        
        edges = self.edges
        edge_id = rand.choice(edges)
        edge = list(self.edges.members(edge_id))
        speaker = rand.choice(edge)
        before_dict = self.count_by_vocab_in_edge(edge)
        #print(list(self.nodes.attrs), '\n')

        if display:
            self.figs.append(self.draw)
        # print(len(self.figs))
        
        if verbose:
            #print(f'{self.edges.members()}')
            print(f'Edge: {edge}')
            print(f'Speaker: {speaker}')
        
        broadcast = random.choice(self.get_attr(speaker, 'vocab'))

        test_stat_beta = np.random.binomial(1, self.get_attr(speaker, 'beta'))
        test_stat_q = np.random.binomial(1, self.get_attr(speaker, 'q'))
        
        if self.rule == 'Unanimous':
            if all([broadcast in self.get_attr(i, 'vocab') for i in edge]):
                # if verbose:
                #     print('agreement possible')
                if test_stat_beta:
                    # if verbose:
                    #     print('beta condition satisfied')
                    for j in edge:
                        if not self.get_attr(j, 'committed'): # sets all listener nodes to vocab=broadcast
                            xgi.classes.function.set_node_attributes(self, {j: {'vocab':[broadcast]}})
                    # print('consensus reached')
                else:
                    pass
            else:
                for j in edge:
                    if broadcast not in self.get_attr(j, 'vocab') and not self.get_attr(j, 'committed'):
                        ### The update below could be improved
                        # print('should add word to vocabs')
                        xgi.classes.function.set_node_attributes(self, \
                        {j: {'vocab': self.get_attr(j, 'vocab') + [broadcast]}})
                if test_stat_q:
                    self.rewire(edge_id, speaker)
                    # print('rewired')
        
                        
        # if self.rule == 'Union':
        #     if sum([1 for i in edge if broadcast in self.get_attr(i, 'vocab')]) > 1:
        #          if test_stat:
        #              for i in edge:
        #                  if not self.get_attr(i, 'committed'): # sets all listener nodes to vocab=broadcast
        #                      xgi.classes.function.set_node_attributes(self, {i:{'vocab':[broadcast]}})
        #              if not self.get_attr(speaker, 'committed'): # sets speaker node to vocab=broadcast
        #                  xgi.classes.function.set_node_attributes(self, {speaker:{'vocab':[broadcast]}})
        #          else:
        #              for i in edge:
        #                  if broadcast not in self.get_attr(i, 'vocab') and not self.get_attr(i, 'committed'):
        #                      self.get_attr(i, 'vocab').append(broadcast)
        #     else:
        #          for i in edge:
        #              if broadcast not in self.get_attr(i, 'vocab') and not self.get_attr(i, 'committed'):
        #                      xgi.classes.function.set_node_attributes(self, \
        #                     {j: {'vocab': self.get_attr(j, 'vocab') + [broadcast]}})
        
        after_dict = self.count_by_vocab_in_edge(edge)
        diff_dict = {}
        
        for i in after_dict.keys():
            diff_dict[i] = after_dict[i]-before_dict[i]
            
        if verbose:
            
            print(f'Word broadcast: {broadcast}')
            print(before_dict)
            print(after_dict)
            print(diff_dict)
            print([i[1]['vocab'] for i in list(self.nodes.attrs)])
            print(list(self.edges.members()))
            print()
            
        return diff_dict

    def rewire(self, edge_id, speaker):
        edge = self.edges.members(edge_id)
        full_edge = edge.copy()
        # print(edge)
        edge.remove(speaker)
        neighbouring_edges = []
        for node in edge:
            neighbouring_edges.append(list(self.nodes.neighbors(node)))
        neighbouring_edges = set(flatten(neighbouring_edges))
        
        for node in full_edge:
            if node in neighbouring_edges:
                neighbouring_edges.remove(node)

        #check there is a unique neighbour to link to
        if len(neighbouring_edges) > 0:
            #remove the speaker from the edge
            # print('there are neighbours')
            self.remove_node_from_edge(edge_id, speaker)
            rand_neighbour = rand.choice(list(neighbouring_edges))
            # print(neighbouring_edges)
            # print(rand_neighbour)
            self.add_node_to_edge(edge_id, rand_neighbour)
        edge = self.edges.members(edge_id)
        self.rewire_counter += 1
        # print(edge)
        return None

    def get_attr(self, node, attr):
        """Function to easily obtain a given attribute from a given node in a hypergraph.

        Args:
            node (int): Integer label of the node being interrogated.
            attr (str): Name of the required attribute.

        Returns:
            (Various types): The requested attribute of the given node.
        """
        return xgi.stats.nodestats.attrs(self, [node], attr)[node]
    
    def count_by_attr(self, attr, val, norm=False):
        if norm:
            return len(self.nodes.filterby_attr(attr, val))/self.num_nodes
        else:
            return len(self.nodes.filterby_attr(attr, val))
            
    def count_by_vocab_in_edge(self, edge):

        count_dict = {'A':0, 'B':0, 'AB':0}

        for i in edge:
            if self.get_attr(i, 'vocab') == ['A'] or self.get_attr(i, 'vocab') == ['B']:
                
                count_dict[self.get_attr(i, 'vocab')[0]] += 1
            elif self.get_attr(i, 'vocab') == ['A', 'B'] or self.get_attr(i, 'vocab') == ['B', 'A']:
                
                count_dict['AB'] += 1
        return count_dict

    def run(self, runlength, verbose=False, display_num=None):
        """runs a complete naming game on a given set of edges

        Args:
            edges (list, ndim=3): list of edges in the hypergraph. axis=0 refers to timesteps, although we don't consider time resolved data yet, so this axis should always have len=1.
                                    Axis=1 is a list of edges, axis=2 is a list of nodes in a given edge
            runlength (int): number of interactions to compute
            verbose (bool, optional): whether or not the status of the interaction should be displayed. Useful for debugging, impractical for large hypergraphs. Defaults to False.

        Returns:
            dict: dictionary containing a list of length=runlength for each possible vocabulary. shows the evolution of the number of agents with a given vocabulary over time.
        """
        if display_num != None:
            frames_to_display = np.linspace(0, runlength-1, endpoint=True, num=display_num, dtype=int).tolist()
            print(frames_to_display)

        vocab_counts = {'A':np.zeros((runlength+1)), 'B':np.zeros((runlength+1)), 'AB':np.zeros((runlength+1)), 'Singleton':np.zeros((runlength+1))}
        vocab_counts['A'][0] = self.count_by_attr('vocab', ['A'], False)
        vocab_counts['B'][0] = self.count_by_attr('vocab', ['B'], False)
        vocab_counts['AB'][0] = self.count_by_attr('vocab', ['A', 'B'], False)+self.count_by_attr('vocab', ['B', 'A'], False)
        vocab_counts['Singleton'][0] = self.nodes.degree.aslist().count(0)
        for i in range(runlength):
            # if i in frames_to_display:
            #     diff_dict = self.interact_and_advance(verbose=verbose, display=True)
            # else:
            #     diff_dict = self.interact_and_advance(verbose=verbose, display=False)
            diff_dict = self.interact_and_advance(verbose=verbose, display=False)
            vocab_counts['A'][i+1] = vocab_counts['A'][i] + diff_dict['A']
            vocab_counts['B'][i+1] = vocab_counts['B'][i] + diff_dict['B']
            vocab_counts['AB'][i+1] = vocab_counts['AB'][i] + diff_dict['AB']
            vocab_counts['Singleton'][i+1] = self.nodes.degree.aslist().count(0)
        final_deg = self.nodes.degree.aslist()
        final_vocab_list = [i[1]['vocab'] for i in list(self.nodes.attrs)]
        return vocab_counts, final_deg, final_vocab_list


def get_edges_and_uniques(fname):
    import json
    with open(fname) as json_file:
        edges = [json.load(json_file)]
        
    edges_flat_1 = list(itertools.chain(*edges))
    edges_flat_2 = list(itertools.chain(*edges_flat_1))
    unique_id = list(set(edges_flat_2))
    return edges, unique_id



def run_ensemble_experiment(prop_committed, beta_non_committed, beta_committed, ensemble_size, run_length, social_structure, rule='Unanimous', thr=3, q_non_committed=0, q_committed=0, display_num=None):
    
    ### this line can be changed depending on which threshold we would like to use, 2 is our data, and data relating to other values come from https://github.com/iaciac/higher-order-NG
    edges, unique_id = get_edges_and_uniques(f'../data/aggr_15min_cliques_thr{thr}_{social_structure}.json')
    ###
    output_fname = f'{social_structure}_{prop_committed}_{beta_non_committed}_{beta_committed}_{q_non_committed}_{q_committed}_{run_length}_{ensemble_size}'
    
    ### This part deletes a file if it already exists
    if os.path.exists(f"outputs/{output_fname}.csv"):
        os.remove(f"outputs/{output_fname}.csv")
    if os.path.exists(f"aux_outputs/{output_fname}.csv"):
        os.remove(f"aux_outputs/{output_fname}.csv")
    ###

    for k in tqdm(range(ensemble_size)):
        H = HigherOrderNamingGame(edges=edges, rule=rule)

        number_committed = round(len(unique_id)*prop_committed)
        rand.shuffle(unique_id)

        committed_nodes, uncommitted_nodes = np.split(np.array(unique_id), [number_committed])
        
        H.add_naming_game_node(uncommitted_nodes, ['A'], False, beta=beta_non_committed, q=q_non_committed)
        
        H.add_naming_game_node(committed_nodes, ['B'], True, beta=beta_committed, q=q_committed)
        
        H.add_edges_from(edges[0])
        initial_deg = H.nodes.degree.aslist()

        with open(f'outputs/{output_fname}.csv', 'a') as f:
            write = csv.writer(f)
            stats, final_deg, final_vocab_list = H.run(run_length, False, display_num=display_num)
            write.writerow(stats['A'])
            write.writerow(stats['B'])
            write.writerow(stats['AB'])
            write.writerow(stats['Singleton'])
        l_conn_comp = xgi.algorithms.connected.largest_connected_component(H)
        conn_comps = xgi.algorithms.connected.connected_components(H)
        conn_comp_sizes = [len(i) for i in conn_comps]
        print(f'size of largest connected component = {len(l_conn_comp)}')
        print(f'sizes of connected components = {conn_comp_sizes}')
        print(f'number of connected components = {xgi.algorithms.connected.number_connected_components(H)}')
        print(f'number of isolates = {len(H.nodes.isolates())}')
        with open(f'aux_outputs/{output_fname}.csv', 'a') as h:
                write = csv.writer(h)
                write.writerow(initial_deg)
                write.writerow([float(H.rewire_counter)])
                write.writerow(final_deg)
                write.writerow(final_vocab_list)
                h.close()
        
        # for i in range(display_num):
        #     fig = H.figs[i]
        #     fig.savefig(f'test{i}.png')
        

print('update19')
# # test code

output_fname = 'test'
edges = [[1,2,3], [1,2], [2,3], [1,4]]
committed_nodes = [3, 4]
uncommitted_nodes = [1,2]

H = HigherOrderNamingGame(rule='Unanimous')
H.add_naming_game_node(uncommitted_nodes, ['A'], False, beta=1, q=1)
H.add_naming_game_node(committed_nodes, ['B'], True, beta=1, q=1)
print(edges[0])
H.add_edges_from(edges)

out = H.run(20, True, display_num=3)

# # run_ensemble_experiment(0.03, 0.27, 0.27, 1, 10**5, 'LyonSchool', q_non_committed=1, q_committed=1)
#%%
