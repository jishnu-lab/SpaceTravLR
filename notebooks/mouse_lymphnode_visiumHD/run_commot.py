import scanpy as sc
import numpy as np
import pandas as pd

import argparse

parser = argparse.ArgumentParser()
parser.add_argument('--sample', type=str, default='1')
args = parser.parse_args()

sample = args.sample


adata = sc.read_h5ad(f'/ix/djishnu/shared/djishnu_kor11/training_data_2025/mouse_lymph{sample}_visiumHD.h5ad')

# COMMOT takes forever to run and is computationally expensive, subsample 20k
np.random.seed(42)
adata = adata[np.random.choice(adata.obs_names, size=20000, replace=False)]

print(adata)
print(adata.obs['cell_type'].value_counts())

import commot as ct 
df_ligrec = ct.pp.ligand_receptor_database(
    database='CellChat', 
    species='mouse', 
    signaling_type=None
)
    
df_ligrec.columns = ['ligand', 'receptor', 'pathway', 'signaling']  

df_ligrec['name'] = df_ligrec['ligand'] + '-' + df_ligrec['receptor']
len(df_ligrec['name'].unique())

import sys
sys.path.append('../../src')
from spaceoracle.tools.network import expand_paired_interactions

expanded = expand_paired_interactions(df_ligrec)
genes = set(expanded.ligand) | set(expanded.receptor)
genes = list(genes)

expanded

expanded = expanded[expanded.ligand.isin(adata.var_names) & expanded.receptor.isin(adata.var_names)]
expanded


ct.tl.spatial_communication(adata,
    database_name='user_database', 
    df_ligrec=expanded, 
    dis_thr=400, 
    heteromeric=False
)

adata.write_h5ad(f'commot/{sample}.h5ad')

expanded['rename'] = expanded['ligand'] + '-' + expanded['receptor']

from tqdm import tqdm

for name in tqdm(expanded['rename'].unique()):

    ct.tl.cluster_communication(adata, database_name='user_database', pathway_name=name, clustering='cell_type',
        random_seed=12, n_permutations=100)
    
from collections import defaultdict
data_dict = defaultdict(dict)

for name in expanded['rename']:
    data_dict[name]['communication_matrix'] = adata.uns[f'commot_cluster-cell_type-user_database-{name}']['communication_matrix']
    data_dict[name]['communication_pvalue'] = adata.uns[f'commot_cluster-cell_type-user_database-{name}']['communication_pvalue']

import pickle
with open(f'/ix/djishnu/shared/djishnu_kor11/commot_outputs/mouse_lymph{sample}_visiumHD_communication.pkl', 'wb') as f:
    pickle.dump(data_dict, f)   


from tqdm import tqdm
import pickle
import commot as ct 

# reload the whole adata (not just the 20k subsample)
adata = sc.read_h5ad(f'/ix/djishnu/shared/djishnu_kor11/training_data_2025/mouse_lymph{sample}_visiumHD.h5ad')
print(adata)

with open(f'/ix/djishnu/shared/djishnu_kor11/commot_outputs/mouse_lymph{sample}_visiumHD_communication.pkl', 'rb') as f:
    info = pickle.load(f)

df_ligrec = ct.pp.ligand_receptor_database(
    database='CellChat', 
    species='mouse', 
    signaling_type=None
)
df_ligrec.columns = ['ligand', 'receptor', 'pathway', 'signaling']  
df_ligrec['name'] = df_ligrec['ligand'] + '-' + df_ligrec['receptor']
len(df_ligrec['name'].unique())

import sys
sys.path.append('../../src')
from spaceoracle.tools.network import expand_paired_interactions
expanded = expand_paired_interactions(df_ligrec)
genes = set(expanded.ligand) | set(expanded.receptor)
genes = list(genes)
expanded = expanded[expanded.ligand.isin(adata.var_names) & expanded.receptor.isin(adata.var_names)]
ct.tl.spatial_communication(adata,
    database_name='user_database', 
    df_ligrec=expanded, 
    dis_thr=400, 
    heteromeric=False
)
expanded['rename'] = expanded['ligand'] + '-' + expanded['receptor']


len(info.keys())
def get_sig_interactions(value_matrix, p_matrix, pval=0.3):
    p_matrix = np.where(p_matrix < pval, 1, 0)
    return value_matrix * p_matrix

interactions = {}
for lig, rec in tqdm(zip(expanded['ligand'], expanded['receptor'])):
    name = lig + '-' + rec

    if name in info.keys():

        value_matrix = info[name]['communication_matrix']
        p_matrix = info[name]['communication_pvalue']

        sig_matrix = get_sig_interactions(value_matrix, p_matrix)
        
        if sig_matrix.sum().sum() > 0:
            interactions[name] = sig_matrix
    
len(interactions)


# create cell x gene matrix
ct_masks = {ct: adata.obs['cell_type'] == ct for ct in adata.obs['cell_type'].unique()}

df = pd.DataFrame(index=adata.obs_names, columns=genes)
df = df.fillna(0)

for name in tqdm(interactions.keys(), total=len(interactions)):
    lig, rec = name.rsplit('-', 1)
    
    tmp = interactions[name].sum(axis=1)
    for ct, val in zip(interactions[name].index, tmp):
        df.loc[ct_masks[ct], lig] += tmp[ct]
    
    tmp = interactions[name].sum(axis=0)
    for ct, val in zip(interactions[name].columns, tmp):
        df.loc[ct_masks[ct], rec] += tmp[ct]

df.shape
print('Number of LR filtered using celltype specificity:')
np.where(df > 0, 1, 0).sum().sum() / (df.shape[0] * df.shape[1])

cell_threshes = df
df.to_csv(f'/ix/djishnu/shared/djishnu_kor11/commot_outputs/mouse_lymph{sample}_visiumHD_cell_threshes.csv')

# import sys 
# sys.path.append('../../src')
# from spaceoracle.models.parallel_estimators import *

# adata.uns['cell_thresholds'] = cell_threshes

# from spaceoracle.oracles import BaseTravLR

# pcs = BaseTravLR.perform_PCA(adata)
# BaseTravLR.impute_clusterwise(adata)

# adata = init_received_ligands(
#     adata, 
#     radius=800, 
#     contact_distance=50, 
#     cell_threshes=cell_threshes
# )

# todelete = [x for x in adata.uns.keys() if 'commot_cluster-cell_type-user_database' in x]
# for key in todelete:
#     del adata.uns[key]
# del adata.obsm['commot-user_database-sum-sender']
# del adata.obsm['commot-user_database-sum-receiver']
# adata
# for key in ['commot-user_database-info']:
#     del adata.uns[key]
    
# adata.write_h5ad(f'/ix/djishnu/shared/djishnu_kor11/training_data_2025/mouse_lymph{sample}_visiumHD.h5ad')