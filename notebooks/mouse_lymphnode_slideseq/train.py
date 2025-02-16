import sys
sys.path.append('../../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
from spaceoracle import SpaceTravLR
from spaceoracle.tools.network import RegulatoryFactory

co_grn = RegulatoryFactory(
    colinks_path='/ix/djishnu/shared/djishnu_kor11/training_data_2025/mLND3-1_v4_colinks.pkl',
    organism='human',
    annot='cell_type_int'
)

adata = sc.read_h5ad(
    '/ix/djishnu/shared/djishnu_kor11/training_data_2025/mLND3-1_v4.h5ad')

print(adata)

star = SpaceTravLR(
    adata=adata,
    annot='cell_type_int', 
    max_epochs=200, 
    learning_rate=5e-4, 
    spatial_dim=64,
    batch_size=512,
    radius=100,
    grn=co_grn,
    save_dir='/ix/djishnu/shared/djishnu_kor11/super_filtered_runs/mLDN3-1_v4'
)

star.run()

exit()