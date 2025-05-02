import sys
sys.path.append('../../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
import pandas as pd
from spaceoracle import SpaceTravLR

from spaceoracle.tools.network import RegulatoryFactory
from spaceoracle.gene_factory import GeneFactory

base_dir = '/ix/djishnu/shared/djishnu_kor11/'

co_grn = RegulatoryFactory(
    colinks_path=base_dir + 'training_data_2025/snrna_human_tonsil_colinks.pkl',
    annot='cell_type_int'
)

adata = sc.read_h5ad(
    base_dir + 'training_data_2025/snrna_human_tonsil.h5ad')


print(adata)

star = SpaceTravLR(
    adata=adata,
    annot='cell_type_int', 
    max_epochs=100, 
    learning_rate=5e-3, 
    spatial_dim=64,
    batch_size=512,
    grn=co_grn,
    radius=400,
    contact_distance=50,
    save_dir=base_dir + 'lasso_runs/human_tonsil'
)

star.run()

gf = GeneFactory.from_json(
    adata=star.adata, 
    json_path=star.save_dir + '/run_params.json', 
)

gf.load_betas()

gf.genome_screen(
    save_to=base_dir + '/genome_screens/human_tonsil',
    n_propagation=4,
    priority_genes=['AICDA', 'BACH2', 'BATF', 'BCL2', 'BCL2A1', 'BCL6', 'BMS1P14',
       'CCL19', 'CCL21', 'CCL5', 'CCR4', 'CCR6', 'CCR7', 'CD19', 'CD274',
       'CD28', 'CD40', 'CD40LG', 'CD80', 'CD83', 'CD86', 'CR2', 'CXCL12',
       'CXCL13', 'CXCL14', 'CXCR4', 'CXCR5', 'EBI3', 'EGR1', 'EGR2',
       'EGR3', 'EPCAM', 'FOXO1', 'FOXP3', 'GATA3', 'ICAM1', 'ICAM2',
       'ICAM3', 'ICOS', 'ICOSLG', 'ID2', 'IL4', 'IL6', 'IL6R', 'IL6ST',
       'IRF4', 'IRF8', 'ITGA5', 'ITGAM', 'ITGB1', 'ITGB2', 'LGALS9',
       'LMO2', 'MICOS10', 'MICOS13', 'NFKB1', 'NFKB2', 'NFKBIA', 'NFKBIB',
       'NFKBID', 'NFKBIE', 'NFKBIL1', 'NFKBIZ', 'PAX5', 'PDCD1', 'PDCD11',
       'PDCD1LG2', 'PRDM1', 'S1PR1', 'S1PR2', 'S1PR3', 'S1PR4', 'SATB1',
       'SDF2', 'SDF2L1', 'SDF4', 'STAT1', 'STAT3', 'STAT4', 'STAT6',
       'TBX21', 'TICAM1', 'TICAM2', 'TRAF3', 'VCAM1']
)

exit()