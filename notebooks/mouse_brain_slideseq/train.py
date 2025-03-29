import sys
sys.path.append('../../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
from spaceoracle import SpaceTravLR
from spaceoracle.tools.network import RegulatoryFactory
from spaceoracle.gene_factory import GeneFactory

base_dir = '/ix1/ylee/kor11/djishnu_kor11/'
fname = 'mouse_brain_wt_slideseq'

co_grn = RegulatoryFactory(
    colinks_path=base_dir + f'training_data_2025/{fname}_colinks.pkl',
    annot='cell_type_int'
)

adata = sc.read_h5ad(
    base_dir + f'training_data_2025/{fname}.h5ad')

print(adata)

star = SpaceTravLR(
    adata=adata,
    annot='cell_type_int', 
    max_epochs=150, 
    learning_rate=5e-3, 
    spatial_dim=64,
    batch_size=512,
    grn=co_grn,
    radius=200,
    contact_distance=30,
    save_dir=base_dir + f'lasso_runs/{fname}'
)

star.run()

gf = GeneFactory.from_json(
    adata=star.adata, 
    json_path=star.save_dir + '/run_params.json', 
)

gf.load_betas()

gf.genome_screen(
    save_to=base_dir + f'/genome_screens/{fname}',
    n_propagation=4
)

exit()