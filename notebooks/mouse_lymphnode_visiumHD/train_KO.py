import sys
sys.path.append('../../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
from spaceoracle import SpaceTravLR
from spaceoracle.tools.network import RegulatoryFactory
from spaceoracle.gene_factory import GeneFactory

part_name = 'd'

base_dir = '/ix/djishnu/shared/djishnu_kor11/'
co_grn = RegulatoryFactory(
    colinks_path=base_dir + 'training_data_2025/mouse_lymphKO4_visiumHD_colinks.pkl',
    annot='cell_type_int'
)
adata = sc.read_h5ad(
    base_dir + f'training_data_2025/mouse_lymphKO4{part_name}_visiumHD.h5ad')

print(adata)

star = SpaceTravLR(
    adata=adata,
    annot='cell_type_int', 
    max_epochs=100, 
    learning_rate=5e-3, 
    spatial_dim=64,
    batch_size=512,
    grn=co_grn,
    radius=800,
    contact_distance=50,
    save_dir=base_dir + f'lasso_runs/mouse_lymphKO4{part_name}_visiumHD'
)

star.run()

