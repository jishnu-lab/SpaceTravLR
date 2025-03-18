import sys
sys.path.append('../../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
from spaceoracle import SpaceTravLR
import pandas as pd
from spaceoracle.tools.network import RegulatoryFactory
from spaceoracle.gene_factory import GeneFactory

base_dir = '/ix/djishnu/shared/djishnu_kor11/'

co_grn = RegulatoryFactory(
    colinks_path=base_dir + 'training_data_2025/mouse_kidney_13_colinks.pkl',
    annot='cell_type_int'
)

adata = sc.read_h5ad(
    base_dir + 'training_data_2025/mouse_kidney_13.h5ad')


cell_threshes = pd.read_parquet(
    base_dir + 'training_data_2025/mouse_kidney_13_LRs.parquet')
adata.uns['cell_thresholds'] = cell_threshes

print(adata)

star = SpaceTravLR(
    adata=adata,
    annot='cell_type_int', 
    max_epochs=100, 
    learning_rate=5e-4, 
    spatial_dim=64,
    batch_size=512,
    grn=co_grn,
    radius=100,
    contact_distance=30,
    save_dir=base_dir + 'lasso_runs/mouse_kidney_13'
)

star.run()

gf = GeneFactory.from_json(
    adata=star.adata, 
    json_path=star.save_dir + '/run_params.json', 
)

gf.load_betas()

gf.genome_screen(
    save_to=base_dir + '/genome_screens/mouse_kidney_13',
    n_propagation=4
)

exit()