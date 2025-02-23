import os
import numpy as np
import pandas as pd 
import scanpy as sc

import sys 
sys.path.append('../../src')
from spaceoracle.prophets import Prophet



outdir = '/ix/djishnu/shared/djishnu_kor11/perturbations/mLDN3-1_v4_Pax5'

adata_train = sc.read_h5ad(
    '/ix/djishnu/shared/djishnu_kor11/training_data_2025/mLND3-1_v4.h5ad')

pro = Prophet(
    adata=adata_train, 
    models_dir='/ix/djishnu/shared/djishnu_kor11/super_filtered_runs/mLDN3-1_v4', 
    annot='cell_type_int', 
    annot_labels='cell_type', 
    radius=100
)

pro.compute_betas()

pro.perturb_batch(
    target_genes=['Bcl6'], 
    n_propagation=2, 
    gene_expr=0, 
    save_to='/ix/djishnu/shared/djishnu_kor11/perturbations/mLDN3-1_v4_Ally'
)

pro.perturb_batch(
    target_genes=['Bcl6'], 
    n_propagation=1, 
    gene_expr=0, 
    save_to='/ix/djishnu/shared/djishnu_kor11/perturbations/mLDN3-1_v4_Ally'
)

exit()