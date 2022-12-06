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


#%%

# model constructor

class HigherOrderNamingGame(xgi.Hypergraph):
    def __init__(self, edges, unique_id, prop_committed, rule='Unanimous', beta_mod_rule=None, beta_mod_rule_params=None, incoming_data=None, **attr):
        xgi.Hypergraph.__init__(self, incoming_data, **attr)
        self.rule = rule
        self.beta_mod_rule = beta_mod_rule
        self.beta_mod_rule_params = beta_mod_rule_params
        self.edge_list = edges[0]

        number_committed = round(len(unique_id)*prop_committed)
        rand.shuffle(unique_id)

        committed_nodes, uncommitted_nodes = np.split(np.array(unique_id), [number_committed])
        
        self.add_naming_game_node(uncommitted_nodes, ['A'], False, beta=None)
        self.add_naming_game_node(committed_nodes, ['B'], True, beta=None)
        
        self.add_edges_from(edges[0])

        self.degree_stat = self.nodes.degree
        self.degree_list = self.degree_stat.aslist()
        self.max_degree = self.degree_stat.max()
        self.mean_degree = self.degree_stat.mean()
        self.std_degree = self.degree_stat.std()
        self.beta_used = []
        # print(self.max_degree)
        # print(self.mean_degree)
        # print(self.std_degree)

        if self.beta_mod_rule == 'n_simple_scale':
            self.beta_list = []
            for node in uncommitted_nodes:
                beta = self.calc_beta_using_simple_scale(node)
                self.beta_list.append(beta)
                xgi.classes.function.set_node_attributes(self, {node: {'beta':beta}})
            for node in committed_nodes:
                beta = self.calc_beta_using_simple_scale(node)
                self.beta_list.append(beta)
                xgi.classes.function.set_node_attributes(self, {node: {'beta':beta}})

        if self.beta_mod_rule == 'n_sigmoid':
            self.beta_list = []
            for node in uncommitted_nodes:
                beta = self.calc_beta_using_sigmoid(node)
                self.beta_list.append(beta)
                xgi.classes.function.set_node_attributes(self, {node: {'beta':beta}})
            for node in committed_nodes:
                beta = self.calc_beta_using_sigmoid(node)
                self.beta_list.append(beta)
                xgi.classes.function.set_node_attributes(self, {node: {'beta':beta}})
        
        if self.beta_mod_rule[0:6] == 'e_homo':
            for node in uncommitted_nodes:
                xgi.classes.function.set_node_attributes(self, {node: {'beta':None}})
            for node in committed_nodes:
                xgi.classes.function.set_node_attributes(self, {node: {'beta':None}})



    def calc_beta_using_simple_scale(self, node):
        degree = self.degree_stat[node]
        return degree/self.max_degree
    def calc_beta_using_sigmoid(self, node):
        degree = self.degree_stat[node]
        steepness = 1/self.std_degree
        return self.trans_sigmoid(degree, self.mean_degree, steepness)
    def trans_sigmoid(self, x, a, k):
        return 1/(1+np.exp(-k*(x-a)))
    def inv_logistic_map(self, x):
        a,b,c,d = 0.04595, 0.09, 0.08073, 0.3957
        return -a*np.log((b/x)-c)+d
    def arcsin_map(self, x):
        a,b,c,d,e = 14.04, 0.2272, -0.3407, 0.1845, -0.1421
        return a*np.arcsin(b*(x**3)+c*(x**2)+d*x+e)+1.99
    def lin_map_with_endpoints(self, x):
        y1, y2 = self.beta_mod_rule_params
        return (y2-y1)*x+y1

    def calc_beta_from_homogeneity(self, edge, broadcast, map='simple'):
        edge_size = len(edge)
        vocabs = []
        for node in edge:
            vocabs.append(self.get_attr(node, 'vocab'))
        counter = 0
        for vocab in vocabs:
            if broadcast in vocab:
                counter += 1
        homogeneity = counter/edge_size
        if map == '_simple':
            return homogeneity
        if map == '_inv_logistic':
            return self.inv_logistic_map(homogeneity)
        if map == '_arcsin':
            return self.arcsin_map(homogeneity)
        if map == '_lin_with_endpoints':
            return self.lin_map_with_endpoints(homogeneity)


    def add_naming_game_node(self, list_nodes, vocab, committed=False, beta=1, meta=None):
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
        
        list_dict = [{'vocab':vocab, 'committed':committed, 'beta':beta}]*N
        self.add_nodes_from(zip(list_nodes, list_dict))
        return None
    
    def clear_edges(self):
        """Remove all edges from the graph without altering any nodes. This is an alteration to the XGI code and was implemented in PR #199."""
        for node in self.nodes:
            self._node[node] = set()
        self._edge.clear()
        self._edge_attr.clear()

    def interact_and_advance(self, edge, verbose=False):
        """Function which carries out one interaction of the naming game on a given edge, and advances to the next frame.

        Args:
            edge (list): list of nodes existing on the selected edge.
            verbose (bool, optional): whether or not the status of the interaction should be displayed. Useful for debugging, impractical for large hypergraphs. Defaults to False.

        Returns:
            dict: dictionary containing the changes made to the counts of each vocabulary, used for counting efficiently.
        """
        
        edge = list(edge)
        speaker = rand.choice(edge)
        before_dict = self.count_by_vocab_in_edge(edge)
        #print(list(self.nodes.attrs), '\n')
        
        if verbose:
            #print(f'{self.edges.members()}')
            print(f'Edge: {edge}')
            print(f'Speaker: {speaker}')
        
        broadcast = random.choice(self.get_attr(speaker, 'vocab'))
        if self.beta_mod_rule[0] == 'n':
            beta = self.get_attr(speaker, 'beta')
            test_stat = np.random.binomial(1, beta)
            self.beta_used.append(beta)
        elif self.beta_mod_rule[0:6] == 'e_homo':
            beta = self.calc_beta_from_homogeneity(edge, broadcast, map=self.beta_mod_rule[6:])
            test_stat = np.random.binomial(1, beta)
            self.beta_used.append(beta)
        else:
            raise(Exception('no valid rule selected'))

        
        if self.rule == 'Unanimous':
            if all([broadcast in self.get_attr(i, 'vocab') for i in edge]):
                if verbose:
                    print('agreement possible')
                if test_stat:
                    if verbose:
                        print('beta condition satisfied')
                    for j in edge:
                        if not self.get_attr(j, 'committed'): # sets all listener nodes to vocab=broadcast
                            xgi.classes.function.set_node_attributes(self, {j: {'vocab':[broadcast]}})
                            
                else:
                    pass
            else:
                for j in edge:
                    if broadcast not in self.get_attr(j, 'vocab') and not self.get_attr(j, 'committed'):
                        ### The update below could be improved
                        xgi.classes.function.set_node_attributes(self, \
                        {j: {'vocab': self.get_attr(j, 'vocab') + [broadcast]}})
        
        after_dict = self.count_by_vocab_in_edge(edge)
        diff_dict = {}
        
        for i in after_dict.keys():
            diff_dict[i] = after_dict[i]-before_dict[i]
            
        if verbose:
            
            print(f'Word broadcast: {broadcast}')
            print(before_dict)
            print(after_dict)
            print(diff_dict)
            print(list(self.nodes.attrs))
            print()
            
        return diff_dict

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

    def run(self, edges, runlength, verbose=False):
        """runs a complete naming game on a given set of edges

        Args:
            edges (list, ndim=3): list of edges in the hypergraph. axis=0 refers to timesteps, although we don't consider time resolved data yet, so this axis should always have len=1.
                                    Axis=1 is a list of edges, axis=2 is a list of nodes in a given edge
            runlength (int): number of interactions to compute
            verbose (bool, optional): whether or not the status of the interaction should be displayed. Useful for debugging, impractical for large hypergraphs. Defaults to False.

        Returns:
            dict: dictionary containing a list of length=runlength for each possible vocabulary. shows the evolution of the number of agents with a given vocabulary over time.
        """
        vocab_counts = {'A':np.zeros((runlength+1)), 'B':np.zeros((runlength+1)), 'AB':np.zeros((runlength+1))}
        vocab_counts['A'][0] = self.count_by_attr('vocab', ['A'], False)
        vocab_counts['B'][0] = self.count_by_attr('vocab', ['B'], False)
        vocab_counts['AB'][0] = self.count_by_attr('vocab', ['A', 'B'], False)+self.count_by_attr('vocab', ['B', 'A'], False)
        
        random_edges = rand.choice(self.edges.members(), size = runlength)
        for i,edge in enumerate(random_edges):
            diff_dict = self.interact_and_advance(edge, verbose=verbose)
            vocab_counts['A'][i+1] = vocab_counts['A'][i] + diff_dict['A']
            vocab_counts['B'][i+1] = vocab_counts['B'][i] + diff_dict['B']
            vocab_counts['AB'][i+1] = vocab_counts['AB'][i] + diff_dict['AB']
        return vocab_counts


def get_edges_and_uniques(fname):
    import json
    with open(fname) as json_file:
        edges = [json.load(json_file)]
        
    edges_flat_1 = list(itertools.chain(*edges))
    edges_flat_2 = list(itertools.chain(*edges_flat_1))
    unique_id = list(set(edges_flat_2))
    return edges, unique_id



def run_ensemble_experiment(prop_committed, ensemble_size, run_length, social_structure, rule='Unanimous', beta_mod_rule=None, beta_mod_rule_params=None, thr=3):
    
    ### this line can be changed depending on which threshold we would like to use, 2 is our data, and data relating to other values come from https://github.com/iaciac/higher-order-NG
    edges, unique_id = get_edges_and_uniques(f'../data/aggr_15min_cliques_thr{thr}_{social_structure}.json')
    ###
    output_fname = f'{social_structure}_{beta_mod_rule}_{beta_mod_rule_params}_{prop_committed}_{run_length}_{ensemble_size}'
    test_fname = 'test'
    
    ### This part deletes a file if it already exists
    if os.path.exists(f"outputs/{output_fname}.csv"):
        os.remove(f"outputs/{output_fname}.csv")
    ###
    ### This part deletes a file if it already exists
    if os.path.exists(f"aux_outputs/{output_fname}.csv"):
        os.remove(f"aux_outputs/{output_fname}.csv")
    ###

    for k in tqdm(range(ensemble_size)):
        H = HigherOrderNamingGame(rule=rule, beta_mod_rule=beta_mod_rule, beta_mod_rule_params=beta_mod_rule_params, edges=edges, unique_id=unique_id, prop_committed=prop_committed)
        with open(f'outputs/{output_fname}.csv', 'a') as f:
            write = csv.writer(f)
            stats = H.run(edges, run_length, False)
            write.writerow(stats['A'])
            write.writerow(stats['B'])
            write.writerow(stats['AB'])
            f.close()
        
        if beta_mod_rule == 'e_homo':
            with open(f'aux_outputs/{output_fname}.csv', 'a') as g:
                write = csv.writer(g)
                write.writerow(H.beta_used)
                g.close()


    
    if beta_mod_rule[0] == 'n':
        with open(f'aux_outputs/{output_fname}.csv', 'a') as h:
            write = csv.writer(h)
            write.writerow(H.degree_list)
            write.writerow([float(H.max_degree)])
            write.writerow(H.beta_list)
            h.close()

print('updated10')

#%%
