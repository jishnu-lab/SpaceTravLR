import pandas as pd 
import numpy as np 
import scanpy as sc 
import seaborn as sns
import matplotlib.pyplot as plt
import argparse

import sys 
sys.path.append('/ix/djishnu/alw399/SpaceOracle/src')
from spaceoracle.tools.network import RegulatoryFactory
from spaceoracle.gene_factory import GeneFactory

parser = argparse.ArgumentParser()
parser.add_argument('--sample', type=str, default='lymph1a')
args = parser.parse_args()

sample = args.sample
print(f'sample: {sample}')

co_grn = RegulatoryFactory(
    colinks_path='/ix/djishnu/shared/djishnu_kor11/training_data_2025/mouse_lymph1_visiumHD_colinks.pkl',
    annot='cell_type_int'
)
adata = sc.read_h5ad(f'/ix/djishnu/shared/djishnu_kor11/training_data_2025/mouse_{sample}_visiumHD.h5ad')
gf = GeneFactory.from_json(
    adata, 
    f'/ix/djishnu/shared/djishnu_kor11/covet_runs/mouse_{sample}_visiumHD/run_params.json', 
    beta_scale_factor=1,
    beta_cap=None,
    co_grn=co_grn
)
gf.load_betas(float16=True, obs_names=None)

transferred = adata[adata.obs['cell_type'] == 'Th2'].obs_names.tolist()

goi = 'Ccr4'
simulated_gex = gf.perturb(
    target='Ccr4',
    n_propagation=4,
    gene_expr=0,
    cells=np.where(gf.adata.obs.index.isin(transferred))[0],
)

import os 
os.makedirs(f'/ix/djishnu/shared/djishnu_kor11/genome_screens/mouse_{sample}_visiumHD_COVET', exist_ok=True)
simulated_gex.to_parquet(f'/ix/djishnu/shared/djishnu_kor11/genome_screens/mouse_{sample}_visiumHD_COVET/{goi}_4n_0x_1der.parquet')

