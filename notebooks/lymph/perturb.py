import numpy as np 
import pandas as pd 
import scanpy as sc 
import sys
import os 

sys.path.append('../../src')
from spaceoracle.prophets import Prophet

adata = sc.read_h5ad('/ix/djishnu/shared/djishnu_kor11/training_data_2025/mLND3-1_v4.h5ad')

pro = Prophet(
    adata=adata, 
    models_dir='/ix/djishnu/shared/djishnu_kor11/super_filtered_runs/mLDN3-1_v4', 
    annot='cell_type_int', 
    annot_labels='cell_type', 
    radius=100
)

pro.compute_betas()

gene_list = ['Cxcr4', 'Pax5', 'Il2ra', 'Bach2', 'Gata3', 'Foxp3', 'Bcl11B', 'Tcf7', 'Runx1', 'Lag3'] + pro.ligands
save_dir = '/ix/djishnu/shared/djishnu_kor11/perturbations/mLDN3-1_v4'

# import os 
# from tqdm import tqdm
# for gene in tqdm(gene_list):

#     file_name = os.path.join(save_dir, f'{gene}.parquet')
    
#     if os.path.exists(file_name):
#         print(f'skipping {gene}')
#         continue

#     if gene not in pro.adata.var_names:
#         print(f'{gene} not in adata')
#         continue

#     print(f'perturbing {gene}')
#     pro.perturb(gene)
#     df = pd.DataFrame(pro.adata.layers['simulated_count'], columns=pro.adata.var_names, index=pro.adata.obs_names)
#     df.to_parquet(file_name)


pro.evaluate(
    perturb_dir='/ix/djishnu/shared/djishnu_kor11/perturbations/mLDN3-1_v4',
    img_dir='/ix/djishnu/shared/djishnu_kor11/results/mLDN3-1_v4',
    gene_list=gene_list
)
