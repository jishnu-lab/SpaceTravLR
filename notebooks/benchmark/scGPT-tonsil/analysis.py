# %%
# %%
import scanpy as sc 
import numpy as np 
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
import os, sys 
sys.path.append('../../src')

# %%
base_dir = '/ix/djishnu/shared/djishnu_kor11/'

adata = sc.read_h5ad(base_dir + 'training_data_2025/snrna_human_tonsil.h5ad')
adata

# %%
del adata.uns['received_ligands']
del adata.uns['received_ligands_tfl']

# Remove pre-processing from COMMOT
adata.uns['cell_thresholds'] = pd.DataFrame(
    index=adata.obs.index, 
    columns=adata.var_names).fillna(1)

adata

# %%
sp_maps = pd.read_parquet('/ix/djishnu/shared/djishnu_kor11/scGPT_outputs/tonsil_embeddings.parquet')
sp_maps = sp_maps.reindex(adata.obs.index, axis=0).values

adata.obsm['scGPT'] = sp_maps
adata.obsm['scGPT'].shape

# %%
import sys 
sys.path.append('../../../src')
from spaceoracle.plotting.cartography import Cartography
from spaceoracle.gene_factory import GeneFactory


# %%
gf = GeneFactory.from_json(
    adata, 
    '/ix/djishnu/shared/djishnu_kor11/scGPT_runs/tonsil/run_params.json', 
)

gf.load_betas(float16=True, obs_names=None)

os.makedirs('/ix/djishnu/shared/djishnu_kor11/genome_screens/human_tonsil_scGPT')


max_expr = adata[:, 'IL21'].layers['imputed_count'].max()
max_expr.item()
gex_out = gf.perturb(
    target='IL21', 
    n_propagation=4,
    gene_expr=max_expr.item()
)
gex_out.to_parquet(
    '/ix/djishnu/shared/djishnu_kor11/genome_screens/human_tonsil_scGPT/IL21_4n_maxx.parquet'
)

# %%
genes = [
    'PAX5', 'BCL6', 'FOXP3', 'GATA3', 'PRDM1', 'FOXO1',
    'PDCD1','IL7R', 'CXCR5', 'CXCR4', 'CCR2',
    'IL7', 'GZMA', 'IL10', 'IL6ST', 'IL4', 'LGALS9',
    'IL21'
]

genes = [g for g in np.unique(genes) if g in adata.var_names]
print(genes)





# # %%
# gf.genome_screen(
#     save_to=base_dir + '/genome_screens/human_tonsil_scGPT',
#     n_propagation=4,
#     priority_genes=genes,
#     # mode='overexpress'
#     mode='knockout'
# )


from tqdm import tqdm
for gene in tqdm(genes):
    gex_out = gf.perturb(
        target=gene, 
        n_propagation=4,
        gene_expr=0
    )

    gex_out.to_parquet(
        f'/ix/djishnu/shared/djishnu_kor11/genome_screens/human_tonsil_scGPT/{gene}_4n_0x.parquet'
    )


