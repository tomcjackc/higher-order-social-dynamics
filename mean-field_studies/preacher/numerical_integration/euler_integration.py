#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Mar  9 15:05:07 2023

@author: Marius
"""


import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
from scipy.stats import binom
import itertools
import multiprocessing
import csv
from numpy import genfromtxt
import os

def count_lists(lst):
    d = {}
    total_lists = len(lst)
    for sublst in lst:
        size = len(sublst)
        if size not in d:
            d[size] = 1
        else:
            d[size] += 1
    for size in d:
        d[size] /= total_lists
    return d

def get_edges_and_uniques(fname):
    import json
    with open(fname) as json_file:
        edges = [json.load(json_file)]
        
    edges_flat_1 = list(itertools.chain(*edges))
    edges_flat_2 = list(itertools.chain(*edges_flat_1))
    unique_id = list(set(edges_flat_2))
    return edges, unique_id


def normalize_array(p, dp):
    # Set all negative elements to 0
    dp[p<10**(-20)] = 0
    p[p<10**(-20)] = 0
    p = p+dp
    # Normalize the array
    norm_arr = p / sum(p)
    
    return norm_arr


def normalize(v):
    return v / sum(v)


class system():
    def __init__(self, dist, beta, f_A_init, f_B_init, f_Bcom_init, t_max, q):
        self.t = 0
        self.t_max = t_max
        self.q = q
        
        self.dist = dist
        if self.dist in ['InVS15', 'LyonSchool', 'SFHH', 'Thiers13']:
            edges, unique_id = get_edges_and_uniques(f'../../../data/aggr_15min_cliques_thr3_{dist}.json')
            self.N = len(unique_id)
        elif type(self.dist) == list:
            self.N = self.dist[1]
            self.gamma = self.dist[2]
            self.dist = self.dist[0]
            print(self.N, self.gamma)
            
        self.possible_n = np.linspace(1, self.N, num=self.N, endpoint=True, dtype=int)
        self.trunc_possible_n = self.possible_n[1:-1]
        self.pi_n_init = np.array([self.pi(n) for n in self.possible_n])
        self.pi_n = self.pi_n_init # this pi_n gets updated at each time step and forms the basis of the custom probability distribution in the pi function
        self.beta = beta
        self.dist = dist
        self.f_A = [f_A_init]+[0]*(self.t_max-1)
        self.f_B = [f_B_init]+[0]*(self.t_max-1)
        self.f_AB = [1-f_A_init-f_B_init-f_Bcom_init]+[0]*(self.t_max-1)
        self.f_Bcom = [f_Bcom_init]*self.t_max
        
        

    def magnetisation(self):
        return self.f_A-self.f_B-self.f_Bcom

    def g(self, n):
        return n/self.N

    def pi(self, n):
        if self.t == 0:
            if self.dist == 'poisson':
                return poisson.pmf(n-1, self.gamma)
            if self.dist == 'exponential':
                return (1/self.gamma)*np.e**((-n-2)/self.gamma)
            if self.dist == 'binomial':
                p = self.gamma/self.N #we take gamma to be the mean of the distribution, so gamma=Np
                return binom.pmf(n-1, self.N-1, p)
            if self.dist == 'Thiers13':
                edges, unique_id = get_edges_and_uniques(f'../../../data/aggr_15min_cliques_thr3_{self.dist}.json')
                self.no_edges = len(edges[0])
                dict_edges = count_lists(edges[0])
                if type(n) == type(np.array([])):
                    return np.array([dict_edges.get(j, 0) for j in n])
                else:
                    return dict_edges.get(n, 0)
            if self.dist == 'SFHH':
                edges, unique_id = get_edges_and_uniques(f'../../../data/aggr_15min_cliques_thr3_{self.dist}.json')
                self.no_edges = len(edges[0])
                dict_edges = count_lists(edges[0])
                if type(n) == type(np.array([])):
                    return np.array([dict_edges.get(j, 0) for j in n])
                else:
                    return dict_edges.get(n, 0)
            if self.dist == 'LyonSchool':
                edges, unique_id = get_edges_and_uniques(f'../../../data/aggr_15min_cliques_thr3_{self.dist}.json')
                self.no_edges = len(edges[0])
                dict_edges = count_lists(edges[0])
                if type(n) == type(np.array([])):
                    return np.array([dict_edges.get(j, 0) for j in n])
                else:
                    return dict_edges.get(n, 0)
            if self.dist == 'InVS15':
                edges, unique_id = get_edges_and_uniques(f'../../../data/aggr_15min_cliques_thr3_{self.dist}.json')
                self.no_edges = len(edges[0])
                dict_edges = count_lists(edges[0])
                
                if type(n) == type(np.array([])):
                    return np.array([dict_edges.get(j, 0) for j in n])
                else:
                    return dict_edges.get(n, 0)
            if self.dist == 'uniform':
                if n == self.gamma: #here we assume that all edges are of size gamma
                    return 1
                else:
                    return 0
        else:
            return self.pi_n[n-1]
        
    def pi_prime(self, n):
        sum = 0
        for j in range(1, self.N+1):
            sum += self.pi(j)*(self.N-j)
        return self.pi(n)*(self.N-n)/sum


    # opinion dynamics
    def w_BAB(self):
        sum = 0
        for n in range(1, self.N+1):
            sum += self.g(n)*self.pi(n)*self.w_BAB_n(n)
        return sum
    
    def w_AAB(self):
        sum = 0
        for n in range(1, self.N+1):
            
            sum += self.g(n)*self.pi(n)*self.w_AAB_n(n)
        return sum
    
    def w_ABA(self):
        sum = 0
        for n in range(1, self.N+1):
            sum += self.g(n)*self.pi(n)*self.w_ABA_n(n)
        return sum
    
    def w_ABB(self):
        sum = 0
        for n in range(1, self.N+1):
            sum += self.g(n)*self.pi(n)*self.w_ABB_n(n)
        return sum

    def w_BAB_n(self, n):# all of those type of func need to be corrected
        return self.beta*self.speaker_says_B_given_B_con_poss_lookingatAB(n)*self.B_consensus_poss_lookingatAB(n,k=1)
    
    def w_AAB_n(self, n):
        return self.beta*self.speaker_says_A_given_A_con_poss_lookingatAB(n)*self.A_consensus_poss_lookingatAB(n, k=1)
    
    def w_ABA_n(self, n):
        return self.speaker_says_B_given_B_con_not_possible_lookingatA(n)
    
    def w_ABB_n(self, n):
        return self.speaker_says_A_given_A_con_not_possible_lookingatB(n)

    def B_consensus_poss_lookingatAB(self, n, k = 0):
        mult = 1
        if self.f_B[self.t]+self.f_AB[self.t]+self.f_Bcom[self.t] >= (n-1)/self.N:
            for i in range(k, n): # this should do the product up to and including the i=n-1 term
                mult = mult*(self.f_B[self.t]+self.f_AB[self.t]+self.f_Bcom[self.t]-(i/self.N))*(self.N/(self.N-i))
            return mult
        else:
            return 0

    def A_consensus_poss_lookingatAB(self, n, k = 0):
        mult = 1
        if self.f_A[self.t]+self.f_AB[self.t] >= (n-1)/self.N:
            for i in range(k,n): # this should do the product up to and including the i=n-1 term
                mult = mult*(self.f_A[self.t]+self.f_AB[self.t]-(i/self.N))*(self.N/(self.N-i))
            return mult
        else:
            return 0
    
    def AB_consensus_poss(self, n, k = 0): #where both A and B consensuses are possible (ie all nodes are AB)
        mult = 1
        if self.f_AB[self.t] >= (n-1)/self.N:
            
            for i in range(k,n): # this should do the product up to and including the i=n-1 term
                mult = mult*(self.f_AB[self.t]-(i/self.N))*(self.N/(self.N-i))
            return mult
        else:
            return 0
    
    def speaker_says_B_given_B_con_poss_lookingatAB(self, n):
        term1 = 1/(2*n)
        term2 = ((n-1)/n)*((self.f_B[self.t]+0.5*self.f_AB[self.t]+self.f_Bcom[self.t])/(self.f_B[self.t]+self.f_AB[self.t]+self.f_Bcom[self.t]))
        return term1+term2
    
    def speaker_says_A_given_A_con_poss_lookingatAB(self, n):
        term1 = 1/(2*n)
        term2 = ((n-1)/n)*((self.f_A[self.t]+0.5*self.f_AB[self.t])/(self.f_A[self.t]+self.f_AB[self.t]))
        return term1+term2
    
    def speaker_says_B_given_B_con_not_possible_lookingatA(self, n):
        return ((n-1)/n)*(self.f_B[self.t]+0.5*self.f_AB[self.t]+self.f_Bcom[self.t])

    def speaker_says_A_given_A_con_not_possible_lookingatB(self, n):
        return ((n-1)/n)*(self.f_A[self.t]+0.5*self.f_AB[self.t])

    def df_A(self):
        df_A = self.w_AAB()*self.f_AB[self.t]-self.w_ABA()*self.f_A[self.t]
        return df_A

    def df_B(self):
        df_B = self.w_BAB()*self.f_AB[self.t]-self.w_ABB()*self.f_B[self.t]
        return df_B
    
    #structural dynamics
    def dpi_n_dt(self):
        # n is a list/array in this function
        norm_term = 1/self.no_edges
        dpi_n_dt = norm_term*(self.w_nnm1()*self.pi_prime(self.trunc_possible_n-1)+self.w_nnp1(self.trunc_possible_n)*self.pi(self.trunc_possible_n+1)-self.w_nm1n(self.trunc_possible_n)*self.pi(self.trunc_possible_n)-self.w_np1n()*self.pi_prime(self.trunc_possible_n))
        #print(self.pi(self.possible_n)[:5])
        return dpi_n_dt
    
    def dpi_1_dt(self):
        norm_term = 1/self.no_edges
        dpi_1_dt = norm_term*(self.w_nnp1(1)*self.pi(2)-self.w_np1n()*self.pi_prime(1))
        #print(self.w_nnp1(1))
        return dpi_1_dt
    
    def dpi_N_dt(self):
        # n is a scalar in this function
        norm_term = 1/self.no_edges
        dpi_N_dt = norm_term*(self.w_nnm1()*self.pi_prime(self.N-1)-self.w_nm1n(self.N)*self.pi(self.N))
        return dpi_N_dt
    
    def P_no_consensus(self, n):
        P = 1- self.A_consensus_poss_lookingatAB(n, k=0)*(1-\
            (self.f_AB[self.t]*0.5/(self.f_A[self.t]+self.f_AB[self.t]))*((n-1)/n))-\
            self.B_consensus_poss_lookingatAB(n,k=0)*(1-\
            (self.f_AB[self.t]*0.5/(self.f_B[self.t]+self.f_AB[self.t]+self.f_Bcom[self.t]))*((n-1)/n)) - \
            self.AB_consensus_poss(n,k=0)*(1-\
            (self.f_AB[self.t]*0.5/(self.f_B[self.t]+self.f_AB[self.t]-self.f_Bcom[self.t]))*((n-1)/n)+\
            (self.f_AB[self.t]*0.5/(self.f_A[self.t]+self.f_AB[self.t]))*((n-1)/n))
        
        return P
    
    def w_nnm1(self):
        sum = 0
        for j in range(1, self.N+1):
            sum += self.pi(j)*self.P_no_consensus(j)
        return sum*self.q
    
    def w_nnp1(self, n):
        if type(n)==type(np.array([])):
            return np.array([self.P_no_consensus(j+1) for j in n])*self.q
        else:
            return self.P_no_consensus(n+1)*self.q
    def w_nm1n(self, n):
        if type(n)==type(np.array([])):
            return np.array([self.P_no_consensus(j) for j in n])*self.q
        else:
            return self.P_no_consensus(n)*self.q
    
    def w_np1n(self):
        sum = 0
        for j in range(1, self.N+1):
            sum += self.pi(j)*self.P_no_consensus(j)
        return sum*self.q
    
    # integrating functions
    def int_1step(self, t):

        df_A_dt = self.df_A()
        df_B_dt = self.df_B()

        dpi_0_dt = self.dpi_1_dt()
        dpi_n_dt = self.dpi_n_dt() #this term will be a list/array
        dpi_N_dt = self.dpi_N_dt()
        
        self.f_A[t] = self.f_A[t-1]+df_A_dt
        self.f_B[t] = self.f_B[t-1]+df_B_dt
        
        self.f_AB[t] = 1-self.f_A[t]-self.f_B[t]-self.f_Bcom[t]       
        self.pi_n = normalize_array(self.pi_n, np.concatenate((np.array([dpi_0_dt]),dpi_n_dt,np.array([dpi_N_dt]))))
        self.pi_n_t[t] = self.pi_n
        
        print(t)
        self.t = t
    
    def integrate(self):
        self.pi_n_t = np.zeros((self.t_max, self.N))
        self.pi_n_t[0] = self.pi_n
        
        for t in range(1, self.t_max):
            self.int_1step(t)





def create_and_integrate(dist, beta, t_max, q, p):
    print(f'beta={beta}, p={p}, q={q}\n')
    output_fname = f'{dist}_{p}_{beta}_{beta}_q={q}_{t_max}'
    sys = system(dist=dist, beta=beta, f_A_init=1-p, f_B_init=0, f_Bcom_init=p, t_max=t_max, q=q)
    sys.integrate()

    ### This part deletes a file if it already exists
    if os.path.exists(f"outputs/{output_fname}.csv"):
        os.remove(f"outputs/{output_fname}.csv")
    if os.path.exists(f"outputs/edge_pdf_{output_fname}.csv"):
        os.remove(f"outputs/edge_pdf_{output_fname}.csv")
    ###
    
    arr = np.array([np.array(sys.f_A), np.array(sys.f_B)+np.array(sys.f_Bcom), np.array(sys.f_AB)]).T
    
    df1 = pd.DataFrame(arr,index = np.linspace(0, t_max, num=t_max, dtype=int, endpoint=False), columns = ['f_A', 'f_B', 'f_AB'])
    df1.to_csv(f'outputs/{output_fname}.csv')
    df2 = pd.DataFrame(sys.pi_n_t.T,index = range(1,sys.N +1), columns =np.linspace(0, t_max, num=t_max, dtype=int, endpoint=False))
    df2.to_csv(f'outputs/edge_pdf_{output_fname}.csv')

def run_multiprocessing_ensemble(prop_committed, betas, run_length, social_structures, qs):
    args = []
    for social_structure in social_structures:
        for p in prop_committed:
            for b in betas:
                for q in qs:
                    p = round(p, 2)
                    b = round(b, 2)
                    args.append((social_structure, b, run_length, q, p))
    
   
    with multiprocessing.Pool() as pool:
        # Use the pool to map the function to the arguments
        pool.starmap(create_and_integrate, args)

def create_csvs_from_outputs(prop_committed, betas, run_length, social_structures, qs):
    Bstar = np.zeros((len(betas), len(prop_committed)))
    Astar = np.zeros((len(betas), len(prop_committed)))
    
    for social_structure in social_structures:
        for q in qs:
            for i, p in enumerate(prop_committed):
                for j, b in enumerate(betas):
                    p = round(p, 2)
                    b = round(b, 2)
                    prop_committed[i] = p
                    betas[j] = b


                    fname = f'{social_structure}_{p}_{b}_{b}_q={q}_{run_length}'
                    
                    data = genfromtxt(f'outputs/{fname}.csv', delimiter=',')
                    
                    A_value = data[0, -1]
                    B_value = data[1, -1]
                    AB_value = data[2, -1]
                    
                    Bstar[j,i] = B_value
                    Astar[j,i] = A_value
                    
                    print(p,b) 
            
            fname = f'{len(prop_committed)}x{len(betas)}_{social_structure}_q={q}_{run_length}'
            df = pd.DataFrame(Bstar, index = betas, columns = prop_committed)
            df.to_csv(f'finished_outputs/heatmap_int_B_res_{fname}.csv')
            df = pd.DataFrame(Astar, index = betas, columns = prop_committed)
            df.to_csv(f'finished_outputs/heatmap_int_A_res_{fname}.csv')


if __name__ == '__main__':
    betas = [0.16, 0.28, 0.36, 0.4, 0.76]
    ps = [0.03]
    qs = [1]
    social_structures = ['InVS15']
    run_length = 10**5
    import warnings
    warnings.filterwarnings("ignore")
    
    run_multiprocessing_ensemble(ps, betas, run_length, social_structures, qs)
    create_csvs_from_outputs(ps, betas, run_length, social_structures, qs)
