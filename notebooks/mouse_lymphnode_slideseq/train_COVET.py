import numpy as np 
import pandas as pd 
import scanpy as sc 
import matplotlib.pyplot as plt 
import seaborn as sns 


base_dir = '/ix/djishnu/shared/djishnu_kor11/training_data_2025/'
adata = sc.read_h5ad(base_dir + 'slideseq_mouse_lymphnode.h5ad')
adata

import sys
sys.path.append('../../src')

from spaceoracle.tools.network import RegulatoryFactory
from spaceoracle.gene_factory import GeneFactory
from spaceoracle.astronomer import Astronaut

co_grn = RegulatoryFactory(
    colinks_path=base_dir + 'slideseq_mouse_lymphnode_colinks.pkl',
    annot='cell_type_int'
)

sp_maps = np.load('/ix/djishnu/shared/djishnu_kor11/covet_outputs/mouse_lymphnode_slideseq/COVET.npy')
feature_key = 'COVET'
adata.obsm['COVET'] = sp_maps
adata

neil = Astronaut(
    adata=adata,
    annot='cell_type_int', 
    max_epochs=200, 
    learning_rate=5e-3, 
    spatial_dim=64,
    batch_size=512,
    grn=co_grn,
    radius=800,
    contact_distance=50,
    save_dir='/ix/djishnu/shared/djishnu_kor11/covet_runs/mouse_lymphnode_slideseq'
)


neil.run(sp_maps_key='COVET')