#%%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sb

betas = np.linspace(0, 1, num=5)
ps = np.linspace(0, 0.2, num=5)
qs = [0]
social_structures = ['LyonSchool']
run_length = 10**4

for social_structure in social_structures:
    
    for q in qs:

        fname_A = f'heatmap_int_A_res_{len(betas)}x{len(ps)}_{social_structure}_q={q}_{run_length}'
        data_A = pd.read_csv(f'finished_outputs/{fname_A}.csv', index_col=0)
        fname_B = f'heatmap_int_B_res_{len(betas)}x{len(ps)}_{social_structure}_q={q}_{run_length}'
        data_B = pd.read_csv(f'finished_outputs/{fname_B}.csv', index_col=0)
        
        
        for p in ps:
            p = round(p,2)
            plt.figure()
            plt.plot(data_A.index.values, data_A[f'{p}'], color='k', linestyle='--', label='A')

            plt.plot(data_B.index.values, data_B[f'{p}'], color='tab:blue', label='B')

            plt.legend(title=r'$x$')
            plt.title(f'New Preacher Model {social_structure} q={q} p={p}')
            plt.xlabel(r'$\beta$')
            plt.ylabel(r'$N^{\ast}_{x}(\beta)$')
            plt.savefig(f'figures/fig3b_{social_structure}_{p}_q={q}_{run_length}.pdf')
            plt.show()
        
        colormap = sb.color_palette(palette="RdBu", n_colors=None, desat=None, as_cmap=True)
        sb.heatmap(data_B-data_A, cbar_kws={'label':r'$n_{B}^{\ast}-n_{A}^{\ast}$'}, cmap=colormap, center=0)
        plt.xlabel('p')
        plt.ylabel(r'$\beta$')
        plt.savefig(f'figures/heatmap_res_{len(betas)}x{len(ps)}_{social_structure}_{q}_{run_length}.pdf')

#%%