#%%
import numpy as np
import scipy as sp
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import poisson
import itertools

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


class system():
    def __init__(self, N, beta, f_A_init, f_B_init, f_Bcom_init, gamma, dist, t_max):
        self.gamma = gamma
        self.N = N
        self.beta = beta
        self.f_A_init = f_A_init
        self.f_B_init = f_B_init
        self.f_AB_init = 1-f_A_init-f_B_init-f_Bcom_init
        self.f_Bcom_init = f_Bcom_init

        self.f_A = [f_A_init]
        self.f_B = [f_B_init]
        self.f_AB = [self.f_AB_init]
        self.f_Bcom = [f_Bcom_init]
        self.dist = dist
        self.t_max = t_max
        self.t = 0

    def magnetisation(self):
        return self.f_A-self.f_B-self.f_Bcom

    def g(self, n):
        return n/self.N

    def pi(self, n):
        dist = self.dist
        if dist == 'poisson':
            return poisson.pmf(n-1, self.gamma)
        if dist == 'exponential':
            return (1/self.gamma)*np.e**((-n-2)/self.gamma)
        if dist == 'Thiers13':
            edges, unique_id = get_edges_and_uniques(f'../../../data/aggr_15min_cliques_thr3_{dist}.json')
           
            dict_edges = count_lists(edges[0])
            dict_edges.get(n, 0)
            return dict_edges.get(n, 0)
        if dist == 'SFHH':
            edges, unique_id = get_edges_and_uniques(f'../../../data/aggr_15min_cliques_thr3_{dist}.json')
            
            dict_edges = count_lists(edges[0])
            dict_edges.get(n, 0)
            return dict_edges.get(n, 0)
        if dist == 'LyonSchool':
            edges, unique_id = get_edges_and_uniques(f'../../../data/aggr_15min_cliques_thr3_{dist}.json')
            
            dict_edges = count_lists(edges[0])
            dict_edges.get(n, 0)
            return dict_edges.get(n, 0)
        if dist == 'InVS15':
            edges, unique_id = get_edges_and_uniques(f'../../../data/aggr_15min_cliques_thr3_{dist}.json')
           
            dict_edges = count_lists(edges[0])
            dict_edges.get(n, 0)
            return dict_edges.get(n, 0)
        if dist == 'binomial':
            return 
        if dist == 'uniform':
            if n == self.gamma:
                return 1
            else:
                return 0
            
    
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

    def w_BAB_n(self, n): # all of those type of func need to be corrected
        return self.beta*self.speaker_says_B_given_B_con_poss()*self.B_consensus_poss(n)
    
    def w_AAB_n(self, n):
        return self.beta*self.speaker_says_A_given_A_con_poss()*self.A_consensus_poss(n)
    
    def w_ABA_n(self, n):
        return self.beta*self.speaker_says_B_given_B_con_not_possible()*(1-self.B_consensus_poss(n))
    
    def w_ABB_n(self, n):
        return self.beta*self.speaker_says_A_given_A_con_not_possible()*(1-self.A_consensus_poss(n))

    def B_consensus_poss(self, n):
        mult = 1
        for i in range(0,n): # this should do the product up to and including the i=n-1 term
            mult = mult*(self.f_B[-1]+self.f_AB[-1]+self.f_Bcom[-1]-(i/self.N))
        return mult

    def A_consensus_poss(self, n):
        mult = 1
        for i in range(0,n): # this should do the product up to and including the i=n-1 term
            mult = mult*(self.f_A[-1]+self.f_AB[-1]-(i/self.N))
        return mult
    
    def speaker_says_B_given_B_con_poss(self):
        return ((self.f_B[-1]+0.5*self.f_AB[-1]+self.f_Bcom[-1])/(self.f_B[-1]+self.f_AB[-1]+self.f_Bcom[-1]))
    
    def speaker_says_A_given_A_con_poss(self):
        return ((self.f_A[-1]+0.5*self.f_AB[-1])/(self.f_A[-1]+self.f_AB[-1]))
    
    def speaker_says_B_given_B_con_not_possible(self):
        return (self.f_B[-1]+0.5*self.f_AB[-1]+self.f_Bcom[-1])

    def speaker_says_A_given_A_con_not_possible(self):
        return (self.f_A[-1]+0.5*self.f_AB[-1])

    def df_A(self):
        df_A = self.w_AAB()*self.f_AB[-1]-self.w_ABA()*self.f_A[-1]
        return df_A

    def df_B(self):
        df_B = self.w_BAB()*self.f_AB[-1]-self.w_ABB()*self.f_B[-1]
        return df_B
    
    def int_1step(self):
        self.f_A.append(self.f_A[-1]+self.df_A())
        self.f_B.append(self.f_B[-1]+self.df_B())
        self.f_Bcom.append(self.f_Bcom[-1])
        self.f_AB.append(1-self.f_B[-1]-self.f_A[-1]-self.f_Bcom[-1])

        self.t += 1
    
    def integrate(self):
        for i in range(0, self.t_max-1):
            self.int_1step()
    
    def scipy_integrate(self):
        def func(f, t):
            #print(t)
            self.t = int(t)
            f_A = f[0]
            f_B = f[1]
            f_Bcom = self.f_Bcom_init
            f_AB = 1-f_A-f_B-f_Bcom

            #print(int(t))
            self.f_A.append(f_A)
            self.f_B.append(f_B)
            self.f_Bcom.append(f_Bcom)
            self.f_AB.append(f_AB)

            df_A_dt = self.w_AAB()*f_AB-self.w_ABA()*f_A
            df_B_dt = self.w_BAB()*f_AB-self.w_ABB()*f_B

            return [df_A_dt, df_B_dt]
        
        res = sp.integrate.odeint(func, [self.f_A_init, self.f_B_init], t=np.linspace(0, self.t_max, num=self.t_max, dtype=int, endpoint=False))
        self.scipy_f_A = res[:, 0]
        self.scipy_f_B = res[:, 1]
        self.scipy_f_Bcom = np.full_like(res[:, 0], self.f_Bcom_init)
        self.scipy_f_AB = np.ones_like(res[:, 0])-self.scipy_f_A-self.scipy_f_B-self.scipy_f_Bcom
        self.scipy_M = self.scipy_f_A-self.scipy_f_B-self.scipy_f_Bcom



'''timings

N=100, beta=1, f_A_init=0.8, f_B_init=0.2, f_Bcom_init=0, gamma=2, t_max=1000:
integrate = 36s
scipy_integrate=5.6s


scipy_integrate seems to be faster, even though it's not written brilliantly

notes so far:
-   N=100, beta=0.4, f_A_init=0.92, f_B_init=0, f_Bcom_init=0.08, gamma=5, t_max=10**5:
    really interesting, seems to be on the edge/critical point of the phase diagram
    also N=100, beta=0.4, f_A_init=0.92, f_B_init=0, f_Bcom_init=0.08, gamma=4.7, t_max=10**5
    and N=100, beta=0.4, f_A_init=0.92, f_B_init=0, f_Bcom_init=0.08, gamma=4.67, t_max=10**5

'''
#%%

p=0.06

sys = system(N=213, beta=0.4, f_A_init=1-p, f_B_init=0, f_Bcom_init=p, gamma=8, dist = 'InVS15', t_max=10**5)
sys.scipy_integrate()

# plt.figure()
# plt.title(f'N={sys.N}, beta={sys.beta}, f_A_init={sys.f_A_init}, f_B_init={sys.f_B_init}, f_Bcom_init={sys.f_Bcom_init}, gamma={sys.gamma}, t_max={sys.t_max}')
# plt.plot(sys.f_A, label='f_A')
# plt.plot(sys.f_B, label='f_B')
# plt.plot(sys.f_AB, label='f_AB')
# plt.plot(sys.f_Bcom, label='f_Bcom')
# plt.legend()


plt.title(f'N={sys.N}, beta={sys.beta}, f_A_init={sys.f_A_init}, f_B_init={sys.f_B_init}, f_Bcom_init={sys.f_Bcom_init}, gamma={sys.gamma}, t_max={sys.t_max}')
plt.plot(sys.scipy_f_A, label='f_A')
plt.plot(sys.scipy_f_B + sys.scipy_f_Bcom, label='f_B')
plt.plot(sys.scipy_f_AB, label='f_AB')
#plt.plot(sys.scipy_f_Bcom, label='f_Bcom')
plt.xscale('log')
plt.legend()
plt.show()

#plt.title(f'N={sys.N}, beta={sys.beta}, f_A_init={sys.f_A_init}, f_B_init={sys.f_B_init}, f_Bcom_init={sys.f_Bcom_init}, gamma={sys.gamma}, t_max={sys.t_max}')
plt.plot(sys.scipy_M, label='Magnetisation')
plt.xscale('log')
plt.ylim((-1,1))
plt.legend()
plt.show()

#%%