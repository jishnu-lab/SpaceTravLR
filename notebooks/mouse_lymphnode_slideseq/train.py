import sys
sys.path.append('../../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
from spaceoracle import SpaceTravLR
from spaceoracle.tools.network import RegulatoryFactory
from spaceoracle.gene_factory import GeneFactory

base_dir = '/ix/djishnu/shared/djishnu_kor11/'

co_grn = RegulatoryFactory(
    colinks_path=base_dir + 'training_data_2025/slideseq_mouse_lymphnode_colinks.pkl',
    annot='cell_type_int'
)

adata = sc.read_h5ad(
    base_dir + 'training_data_2025/slideseq_mouse_lymphnode.h5ad')

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
    save_dir=base_dir + 'lasso_runs/slideseq_mouse_lymphnode'
)

star.run()

gf = GeneFactory.from_json(
    adata=star.adata, 
    json_path=star.save_dir + '/run_params.json', 
)

gf.load_betas()

gf.genome_screen(
    save_to=base_dir + '/genome_screens/slideseq_mouse_lymphnode',
    n_propagation = 4,
    priority_genes = [
        'Il2', 'Il4',  'Il6st', 'Gzma', 'Il2ra', 
        'Cxcr4', 'Ccr4', 'Il4ra', 'Gata3', 'Gata2', 
        'Pax5', 'Stat4', 'Foxp3', 'Bcl6'
    ]
)

exit()