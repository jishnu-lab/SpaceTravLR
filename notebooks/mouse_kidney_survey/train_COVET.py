import scanpy as sc 
import numpy as np 
import pandas as pd 

import sys
sys.path.append('../../src')

from spaceoracle.tools.network import RegulatoryFactory
from spaceoracle.gene_factory import GeneFactory
from spaceoracle.astronomer import Astronaut


base_dir = '/ix/djishnu/shared/djishnu_kor11/'

co_grn = RegulatoryFactory(
    colinks_path=base_dir + 'training_data_2025/mouse_kidney_13_colinks.pkl',
    annot='cell_type_int'
)

adata = sc.read_h5ad('/ix/djishnu/shared/djishnu_kor11/training_data_2025/mouse_kidney_13.h5ad')


sp_maps = np.load('/ix/djishnu/shared/djishnu_kor11/covet_outputs/mouse_kidney_13/COVET.npy')
feature_key = 'COVET'
adata.obsm['COVET'] = sp_maps

cell_threshes = pd.read_parquet(
    base_dir + 'training_data_2025/mouse_kidney_13_LRs.parquet')
adata.uns['cell_thresholds'] = cell_threshes

neil = Astronaut(
    adata=adata,
    annot='cell_type_int', 
    max_epochs=200, 
    learning_rate=5e-3, 
    spatial_dim=64,
    batch_size=512,
    grn=co_grn,
    radius=200,
    contact_distance=30,
    save_dir=base_dir + 'covet_runs/mouse_kidney_13'
)

neil.run(sp_maps_key='COVET')


# gf = GeneFactory.from_json(
#     adata=adata, 
#     json_path=base_dir + 'covet_runs/mouse_kidney_13' + '/run_params.json', 
# )

# gf.load_betas()
# gf.perturb(target='Mif', n_propagation=4, save_layer=True)
# pd.DataFrame(
#     gf.adata.layers['Mif_4n_0x'],
#     index=gf.adata.obs_names,
#     columns=gf.adata.var_names
# ).to_parquet('/ix/djishnu/shared/djishnu_kor11/genome_screens/mouse_kidney_13_COVET/Mif_4n_0x.parquet')


# gf.genome_screen(
#     save_to=base_dir + '/genome_screens/mouse_kidney_13_COVET',
#     n_propagation=4
# )