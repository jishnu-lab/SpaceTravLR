# %%

# %%
import scanpy as sc 
import numpy as np
import pandas as pd
import os, sys

# %%
import celloracle as co
import matplotlib.pyplot as plt


# %%
adata = sc.read_h5ad('/ix/djishnu/shared/djishnu_kor11/training_data_2025/mLND3-1_v4.h5ad')
adata

# %%
oracle = co.load_hdf5("/ix/djishnu/shared/djishnu_kor11/co_objects/mLND3-1_v4.celloracle.oracle")
links = co.load_hdf5(file_path="/ix/djishnu/shared/djishnu_kor11/co_objects/mLND3-1_v4-links.celloracle.links")

# %%
# tfs = []
# for cluster, df in links.links_dict.items():
#     tfs.extend(df["source"].unique())

# tfs = sorted(set(tfs))
# len(tfs)

# %%
tf = sys.argv[1]

os.makedirs(f'/ix/djishnu/shared/djishnu_kor11/co_results/mLDN3-1_v4/{tf}', exist_ok=True)
oracle.simulate_shift(perturb_condition={tf: 0})

# %%
import json
with open('../../data/GSEA/m2.all.v2024.1.Mm.json', 'r') as f:
    gsea_modules = json.load(f)

# %%
import sys
sys.path.append('../../src')
from spaceoracle.plotting.gsea import * 

gsea_scores = compute_gsea_scores(oracle.adata, gsea_modules)
gsea_scores_perturbed = compute_gsea_scores(oracle.adata, gsea_modules, layer='simulated_count')

# %%
delta_gsea_scores = gsea_scores_perturbed - gsea_scores
delta_gsea_scores.dropna(inplace=True)

delta_gsea_scores['abs_mean'] = delta_gsea_scores.iloc[:, :-1].apply(lambda row: np.abs(row.mean()), axis=1)
delta_gsea_scores.sort_values(by = 'abs_mean', ascending=False, inplace=True)

delta_gsea_scores = delta_gsea_scores.loc[delta_gsea_scores.index.str.contains('BIOCARTA')]
delta_gsea_scores

# %%
show_gsea_scores(
    oracle.adata, delta_gsea_scores, annot='cell_type', n_show=9, 
    savepath=f'/ix/djishnu/shared/djishnu_kor11/co_results/mLDN3-1_v4/{tf}/gsea.png')

# %%
from spaceoracle.plotting.transitions import contour_shift
from spaceoracle.judges import Judge, permute_rows_nsign
import sys

judger = Judge(adata, annot='cell_type')


# %%
results_dir = '/ix/djishnu/shared/djishnu_kor11/co_results/mLDN3-1_v4'
st_dir = '/ix/djishnu/shared/djishnu_kor11/perturbations/mLDN3-1_v4'

os.makedirs(f'{results_dir}/{tf}', exist_ok=True)

# %%
seed=1334

fig, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})
axs.flatten()
contour_shift(oracle.adata, title=f'Cell Identity Shift from {tf} KO', annot='cell_type', seed=seed, ax=axs[0])

delta_X_rndm = oracle.adata.layers['delta_X'].copy()
permute_rows_nsign(delta_X_rndm)
fake_simulated_count = oracle.adata.layers['imputed_count'] + delta_X_rndm

contour_shift(oracle.adata, title=f'Randomized Effect of {tf} KO Shift', annot='cell_type', seed=seed, ax=axs[1], perturbed=fake_simulated_count)

plt.tight_layout()
plt.savefig(f'{results_dir}/{tf}/contour.png')

# %%
nt = pd.DataFrame(oracle.adata.layers['imputed_count'], columns=oracle.adata.var_names)
co = pd.DataFrame(oracle.adata.layers['simulated_count'], columns=oracle.adata.var_names)
st = pd.read_parquet(f'{st_dir}/{tf}.parquet')

co.to_parquet(f'{results_dir}/{tf}/simulated_count.parquet')

# %%
co_sim_adata = judger.create_sim_adata(nt.values, co.values)
co_deg_df = judger.get_expected_degs(co_sim_adata, tf, show=30, save_path=results_dir+f'/{tf}/degs_CO.png')

# %%
st_sim_adata = judger.create_sim_adata(nt.values, st.values)
st_deg_df = judger.get_expected_degs(st_sim_adata, tf, show=30, save_path=results_dir+f'/{tf}/degs_ST.png')

# %%
judger.plot_delta_corr(nt=nt, co=co, pred=st, ko=tf, save_path=results_dir+f'/{tf}/delta_corr.png')

# %%


# %%


# %%



exit()