import sys
sys.path.append('../../src')
import warnings
warnings.filterwarnings("ignore", category=FutureWarning)
import scanpy as sc
from spaceoracle import SpaceTravLR

from spaceoracle.tools.network import RegulatoryFactory


co_grn = RegulatoryFactory(
    colinks_path='/ix/djishnu/shared/djishnu_kor11/training_data_2025/snrna_human_tonsil_colinks.pkl',
    organism='human',
    annot='cell_type_int'
)

adata_train = sc.read_h5ad(
    '/ix/djishnu/shared/djishnu_kor11/training_data_2025/snrna_human_tonsil.h5ad')

print(adata_train)

star = SpaceTravLR(
    adata=adata_train,
    annot='cell_type_int', 
    max_epochs=200, 
    learning_rate=5e-4, 
    spatial_dim=64,
    batch_size=512,
    test_mode=False,
    grn=co_grn,
    radius=250,
    save_dir='/ix/djishnu/shared/djishnu_kor11/models_snrna_human_tonsil_v3'
)

star.run()

exit()