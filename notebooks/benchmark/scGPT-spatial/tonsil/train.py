import scanpy as sc 
import numpy as np 
import pandas as pd 
import matplotlib.pyplot as plt

import sys
sys.path.append('/ix3/djishnu/alw399/SpaceOracle/src')

from spaceoracle.tools.network import RegulatoryFactory
from spaceoracle.gene_factory import GeneFactory

import json 
base_dir = '/ix/djishnu/shared/djishnu_kor11/'

adata = sc.read_h5ad(base_dir + 'training_data_2025/snrna_human_tonsil.h5ad')


with open(base_dir + 'scGPT_outputs/tonsil_mgs_spatial.json', 'r') as f:
    grn = json.load(f)

sp_maps = pd.read_parquet('/ix/djishnu/shared/djishnu_kor11/scGPT_outputs/tonsil_embeddings_spatial.parquet')
sp_maps = sp_maps.reindex(adata.obs.index, axis=0).values

feature_key = 'scGPT_spatial'
adata.obsm[feature_key] = sp_maps

from spaceoracle.astronomer import GeneGeneAstronaut
neil = GeneGeneAstronaut(
    adata=adata,
    annot='cell_type_int', 
    max_epochs=200, 
    learning_rate=5e-3, 
    # spatial_dim=64, # used to create the spatial maps
    batch_size=512,
    grn=grn,
    radius=400,
    contact_distance=50,
    save_dir=base_dir + 'scGPT_runs/tonsil_spatial'
)

neil.run(sp_maps_key=feature_key) 