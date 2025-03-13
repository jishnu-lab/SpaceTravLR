import sys
sys.path.append('../../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
from spaceoracle import SpaceTravLR
import pandas as pd
from spaceoracle.tools.network import RegulatoryFactory


co_grn = RegulatoryFactory(
    colinks_path='/ix/djishnu/shared/djishnu_kor11/training_data_2025/mouse_kidney_13_colinks.pkl',
    annot='cell_type_int'
)

adata = sc.read_h5ad(
    '/ix/djishnu/shared/djishnu_kor11/training_data_2025/mouse_kidney_13.h5ad')


cell_threshes = pd.read_parquet(
    '/ix/djishnu/shared/djishnu_kor11/training_data_2025/mouse_kidney_13_LRs.parquet')
adata.uns['cell_thresholds'] = cell_threshes

print(adata)

star = SpaceTravLR(
    adata=adata,
    annot='cell_type_int', 
    max_epochs=100, 
    learning_rate=5e-3, 
    spatial_dim=64,
    batch_size=512,
    grn=co_grn,
    radius=200,
    contact_distance=30,
    save_dir='/ix/djishnu/shared/djishnu_kor11/lasso_runs/mouse_kidney_13'
)

star.run()

exit()