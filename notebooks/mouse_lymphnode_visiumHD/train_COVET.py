import scanpy as sc 
import numpy as np 
import pandas as pd 

import sys
sys.path.append('../../src')

from spaceoracle.tools.network import RegulatoryFactory
from spaceoracle.gene_factory import GeneFactory
from spaceoracle.astronomer import Astronaut


part = '1d'
base_dir = '/ix/djishnu/shared/djishnu_kor11/'

co_grn = RegulatoryFactory(
    colinks_path=base_dir + f'training_data_2025/mouse_lymph1_visiumHD_colinks.pkl',
    annot='cell_type_int'
)

adata = sc.read_h5ad(f'/ix/djishnu/shared/djishnu_kor11/training_data_2025/mouse_lymph{part}_visiumHD.h5ad')

sp_maps = np.load(f'/ix/djishnu/shared/djishnu_kor11/covet_outputs/mouse_lymphnode_visiumHD/mouse_lymph{part}_visiumHD_COVET.npy')
feature_key = 'COVET'
adata.obsm['COVET'] = sp_maps


neil = Astronaut(
    adata=adata,
    annot='cell_type_int', 
    max_epochs=100, 
    learning_rate=5e-3, 
    spatial_dim=64,
    batch_size=512,
    grn=co_grn,
    radius=800,
    contact_distance=50,
    save_dir=base_dir + f'covet_runs/mouse_lymph{part}_visiumHD'
)

neil.run(sp_maps_key='COVET')

