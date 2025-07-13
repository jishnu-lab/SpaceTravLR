import sys
sys.path.append('../../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
from spaceoracle import SpaceTravLR
from spaceoracle.tools.network import RegulatoryFactory
from spaceoracle.gene_factory import GeneFactory

#################
### Subsample ###
#################

from spaceoracle.models.parallel_estimators import init_received_ligands
import numpy as np
from spaceoracle.models.spatial_map import xyc2spatial_fast
from spaceoracle.models.parallel_estimators import create_spatial_features


sample = 'KO4'
base_dir = '/ix/djishnu/shared/djishnu_kor11/'

co_grn = RegulatoryFactory(
    colinks_path=base_dir + f'training_data_2025/mouse_lymph{sample}_visiumHD_colinks.pkl',
    annot='cell_type_int'
)

adata = sc.read_h5ad(
    base_dir + f'training_data_2025/mouse_lymph{sample}_visiumHD.h5ad')

del adata.obsm['spatial_maps']
del adata.obsm['spatial_features']
del adata.uns['received_ligands']
del adata.uns['received_ligands_tfl']


# we need tto split into batches, it's impossible to train this many cells in one go
batch_size = adata.n_obs // 4
pool = set(adata.obs_names)

parta = np.random.choice(list(pool), size=min(len(pool), batch_size), replace=False)
parta_set = set(parta)

partb = np.random.choice(list(pool - parta_set), size=min(len(pool - parta_set), batch_size), replace=False)
partb_set = set(partb)

partc = np.random.choice(list(pool - parta_set - partb_set), size=min(len(pool - parta_set - partb_set), batch_size), replace=False)
partc_set = set(partc)

partd_set = pool - parta_set - partb_set - partc_set
partd = list(partd_set)

adata_pta = adata[parta].copy()
adata_ptb = adata[partb].copy()
adata_ptc = adata[partc].copy()
adata_ptd = adata[partd].copy()

used = (
    set(adata_ptd.obs_names)
    | set(adata_ptc.obs_names)
    | set(adata_ptb.obs_names)
    | set(adata_pta.obs_names)
)

assert set(pool) == set(used)

radius=800
contact_distance=50
spatial_dim = 64

for part_name, adata in zip(
                    ['a', 'b', 'c', 'd'], 
                    [adata_pta, adata_ptb, adata_ptc, adata_ptd]):
    
    adata = init_received_ligands(
        adata,
        radius=radius, 
        contact_distance=contact_distance, 
        cell_threshes=adata.uns['cell_thresholds']
    )

    xy = np.array(adata.obsm['spatial'])
    cluster_annot = 'cell_type_int'
    cluster_labels = np.array(adata.obs[cluster_annot])

    adata.obsm['spatial_maps'] = xyc2spatial_fast(
        xyc = np.column_stack([xy, cluster_labels]),
        m=spatial_dim,
        n=spatial_dim,
    )

    adata.obsm['spatial_features'] = create_spatial_features(
        adata.obsm['spatial'][:, 0], 
        adata.obsm['spatial'][:, 1], 
        adata.obs['cell_type_int'], 
        adata.obs.index,
        radius=radius
    )

    adata.write_h5ad(base_dir + f'training_data_2025/mouse_lymph{sample}{part_name}_visiumHD.h5ad')


###################
### Full sample ###
###################

# sample = 'KO4'
# base_dir = '/ix/djishnu/shared/djishnu_kor11/'
# adata = sc.read_h5ad(
#     base_dir + f'training_data_2025/mouse_lymph{sample}_visiumHD.h5ad')

# radius=600
# contact_distance=40

# from spaceoracle.models.parallel_estimators import init_received_ligands

# adata = init_received_ligands(
#     adata,
#     radius=radius, 
#     contact_distance=contact_distance, 
#     cell_threshes=adata.uns['cell_thresholds']
# )

# import numpy as np
# from spaceoracle.models.spatial_map import xyc2spatial_fast

# spatial_dim = 64
# xy = np.array(adata.obsm['spatial'])
# cluster_annot = 'cell_type_int'
# cluster_labels = np.array(adata.obs[cluster_annot])

# adata.obsm['spatial_maps'] = xyc2spatial_fast(
#     xyc = np.column_stack([xy, cluster_labels]),
#     m=spatial_dim,
#     n=spatial_dim,
# )

# from spaceoracle.models.parallel_estimators import create_spatial_features

# adata.obsm['spatial_features'] = create_spatial_features(
#     adata.obsm['spatial'][:, 0], 
#     adata.obsm['spatial'][:, 1], 
#     adata.obs['cell_type_int'], 
#     adata.obs.index,
#     radius=radius
# )

# adata.write_h5ad(base_dir + f'training_data_2025/mouse_lymph{sample}_visiumHD.h5ad')

