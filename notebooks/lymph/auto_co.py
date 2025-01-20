import scanpy as sc 
import numpy as np
import pandas as pd
import os

import celloracle as co


adata = sc.read_h5ad('/ix/djishnu/shared/djishnu_kor11/training_data/day3_lymph_rep_1.h5ad')

oracle = co.load_hdf5("../../data/lymph_day3.celloracle.oracle")
links = co.load_hdf5(file_path="../../data/links.celloracle.links")

tfs = []
for cluster, df in links.links_dict.items():
    tfs.extend(df["source"].unique())

tfs = set(tfs)

#################


import sys 
sys.path.append('../../src')

from spaceoracle.prophets import Prophet

pythia = Prophet(
    adata=adata,
    models_dir='/ix/djishnu/shared/djishnu_kor11/models_v2',
    annot='rctd_cluster',
    annot_labels='rctd_celltypes'
)

pythia.compute_betas()


#################

from spaceoracle.judges import Judge
judger = Judge(adata)


results_dir = '/ix/djishnu/shared/djishnu_kor11/results/lymph'
co_savedir = results_dir + '/co_simulated'
st_savedir = results_dir + '/st_simulated'


for tf in tfs:
    finished = os.listdir('/ix/djishnu/alw399/SpaceOracle/notebooks/lymph/finished') 
    if tf in finished:
        continue
    with open('/ix/djishnu/alw399/SpaceOracle/notebooks/lymph/finished/' + tf, 'w') as f:
        f.write('')

    print(tf)

    pythia.perturb(tf)
    np.save(st_savedir + f'/{tf}', pythia.adata.layers['simulated_count'])

    oracle.simulate_shift(perturb_condition={tf: 0})
    np.save(co_savedir + f'/{tf}', oracle.adata.layers['simulated_count'])

    st_sim_adata = judger.create_sim_adata(
        pythia.adata.layers['imputed_count'], 
        pythia.adata.layers['simulated_count']
    )

    st_deg_df = judger.get_expected_degs(st_sim_adata, ko=tf, show=30, save_path=results_dir + f'/comparison/degs/{tf}_ST.png')

    co_sim_adata = judger.create_sim_adata(
        oracle.adata.layers['imputed_count'], 
        oracle.adata.layers['simulated_count']
    )

    co_deg_df = judger.get_expected_degs(co_sim_adata, tf, show=30, save_path=results_dir + f'/comparison/degs/{tf}_CO.png')


    nt_co = pd.DataFrame(oracle.adata.layers['imputed_count'], columns=oracle.adata.var_names)
    co_gex = pd.DataFrame(oracle.adata.layers['simulated_count'], columns=oracle.adata.var_names)
    nt_st = pd.DataFrame(pythia.adata.layers['imputed_count'], columns=oracle.adata.var_names)
    st = pd.DataFrame(pythia.adata.layers['simulated_count'], columns=oracle.adata.var_names)

    judger.plot_delta_corr(nt_co=nt_co, co=co_gex, nt_st=nt_st, pred=st, ko=tf, save_path=results_dir + f'/comparison/delta_corr/{tf}.png')