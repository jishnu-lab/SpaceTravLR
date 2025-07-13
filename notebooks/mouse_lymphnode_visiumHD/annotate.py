# %%
import scanpy as sc
import pandas as pd 
import numpy as np
import seaborn as sns 
import os 
import matplotlib.pyplot as plt

# %% [markdown]
# ### Reload

def annotate(sample):
        
    # %%
    adata_vis = sc.read_h5ad(f'results/lymph_nodes_KO/cell2location_map_{sample}/sp_{sample}.h5ad')
    adata_vis

    # %%
    # sample = 1
    # adata_vis = sc.read_h5ad(f'results/lymph_nodes_analysis/cell2location_map_{sample}/sp_{sample}.h5ad')
    # adata_vis

    # %%
    # sample = 1
    # adata_vis = sc.read_h5ad(f'results_spots/lymph_nodes_analysis/cell2location_map_{sample}/sp_{sample}.h5ad')
    # adata_vis

    # %%
    # sample = 1
    # adata_vis = sc.read_h5ad(f'results_spots/lymph_nodes_KO/cell2location_map_{sample}/sp_{sample}.h5ad')
    # adata_vis

    # %%
    # Count cells with different marker combinations
    ccr4_cells = adata_vis.obs_names[adata_vis[:, 'Ccr4'].X.toarray().flatten() > 0]
    prdm1_cells = adata_vis.obs_names[adata_vis[:, 'Prdm1'].X.toarray().flatten() > 0]
    ccr4_prdm1_cells = set(ccr4_cells) & set(prdm1_cells)
    ccr4_neg_cells = set(adata_vis.obs_names) - set(ccr4_cells)
    prdm1_only_cells = set(prdm1_cells) - set(ccr4_cells)

    # Create dataframe with cell counts
    cell_counts = pd.DataFrame({
        'Cell Type': ['Ccr4+', 'Ccr4-', 'Prdm1+', 'Ccr4+ Prdm1+', 'Ccr4- Prdm1+'],
        'Count': [len(ccr4_cells), len(ccr4_neg_cells), len(prdm1_cells), len(ccr4_prdm1_cells), len(prdm1_only_cells)]
    })


    # %%
    slideseq_genes = sc.read_h5ad('/ix/djishnu/shared/djishnu_kor11/training_data_2025/slideseq_mouse_lymphnode.h5ad').var_names
    shared_genes = set(slideseq_genes) & set(adata_vis.var_names)
    adata_vis = adata_vis[:, list(shared_genes)]
    adata_vis

    # %%
    sc.pl.spatial(
        adata_vis, 
        color=['B', 'CD8+ T', 'DC', 'NK', 'Resting T', 'Tfh', 'Th2', 'Treg', 'gd T'], 
        spot_size=20
    )

    # %%
    adata_vis.layers['raw_count'] = adata_vis.X.copy()
    adata_vis.X.min(), adata_vis.X.max()

    # %%
    adata = adata_vis

    # %%
    adata.var_names = adata.var_names.str.capitalize()

    adata = adata[:, ~adata.var_names.str.contains('Rik')]
    adata = adata[:, ~adata.var_names.str.contains('rik')]
    adata = adata[:, ~adata.var_names.str.contains(r'^Hb\w+-\w+$')]
    adata = adata[:, ~adata.var_names.str.contains('Hp')]
    adata = adata[:, ~adata.var_names.str.startswith('Rp')]
    adata = adata[:, ~adata.var_names.str.startswith('n-r5s')]
    adata = adata[:, ~adata.var_names.str.startswith('n-r5')]
    adata = adata[:, ~adata.var_names.str.startswith('N-r5s')]
    adata = adata[:, ~adata.var_names.str.startswith('N-r5')]
    adata = adata[:, ~adata.var_names.str.startswith('n-R5s')]
    adata = adata[:, ~adata.var_names.str.startswith('n-R5')]
    adata = adata[:, ~adata.var_names.str.startswith('N-R5s')]
    adata = adata[:, ~adata.var_names.str.startswith('N-R5')]
    adata = adata[:, ~adata.var_names.str.startswith('Aa')]
    adata = adata[:, ~adata.var_names.str.startswith('Ab')]
    adata = adata[:, ~adata.var_names.str.startswith('Ac')]
    adata = adata[:, ~adata.var_names.str.startswith('Gm')]
    adata = adata[:, ~adata.var_names.str.startswith('Mir')]
    adata = adata[:, adata.var.index.str.len() > 1]
    adata = adata[:, [i for i in adata.var_names if not (i[:2].isupper() and i[:2].isalpha())]]
    adata = adata[:, [gene for gene in adata.var_names if not gene[-4:].isdigit()]]

    # %%
    sc.pp.normalize_total(adata, inplace=True)
    sc.pp.log1p(adata)

    adata.layers['normalized_count'] = adata.X.copy()

    # %%
    cell_types = ['B', 'CD8+ T', 'DC', 'NK', 'Resting T', 'Tfh', 'Th2', 'Treg', 'gd T']
    df = adata.obs[cell_types]
    print(df.shape)
    df.head(3)


    # %%
    adata_ref= sc.read_h5ad('/ix/djishnu/shared/djishnu_kor11/rctd_outputs/mouse_lymphnode_slideseq/zhongli_ref_202401203_mannually_woDoublet.h5ad')

    abundance = adata_ref.obs['cell_type'].value_counts() / adata_ref.n_obs
    abundance

    # %%
    import copy

    df_thresholded = copy.deepcopy(df)

    for ct in adata_ref.obs['cell_type'].unique():
        threshold = df[ct].quantile(1-abundance[ct])
        print(ct, '|', threshold)
        df_thresholded[ct] = df[ct].apply(lambda x: x if x >= threshold else 0)


    # %%
    # Get primary annotations (cell type with highest value)
    primary_annot = df_thresholded.idxmax(axis=1)

    # Get secondary annotations (cell type with second highest value)
    secondary_annot = df_thresholded.apply(
        lambda x: x.nlargest(2).index[1] if x.nlargest(2).iloc[1] > 0 else None, 
        axis=1
    )

    tertiary_annot = df_thresholded.apply(
        lambda x: x.nlargest(3).index[2] if x.nlargest(3).iloc[2] > 0 else None, 
        axis=1
    )

    adata.obs['primary_annot'] = primary_annot
    adata.obs['secondary_annot'] = secondary_annot
    adata.obs['tertiary_annot'] = tertiary_annot

    # %%
    primary_annot.value_counts()

    # %%
   
    # %%
    # adata = adata_[rules]
    # adata.obs['cell_type'] = adata.obs['split_annot']
    adata.obs['cell_type'] = adata.obs['primary_annot']

    # %%
    sc.pp.pca(adata, n_comps=50)

    try:
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)

    except:
        pass
    # sc.pl.umap(adata, color='cell_type')

    # %%
    # sc.tl.leiden(adata, resolution=0.3)
    # sc.pl.umap(adata, color='leiden')

    # %%
    # keep_leiden = [l for l, c in adata.obs.leiden.value_counts().items() if c > 5]

    # sc.pl.dotplot(adata[adata.obs.leiden.isin(keep_leiden)], 
    #     var_names=var_names, swap_axes=False,
    #     groupby='leiden')

    # %%
    adata

    # %%
    adata.obs.head(3)

    # %%
    # for k in ['primary_annot', 'secondary_annot', 'tertiary_annot', 'split_annot']:
    #     del adata.obs[k]

    for k in ['n_genes_by_counts', 'log1p_n_genes_by_counts', 'total_counts', 'log1p_total_counts', 'pct_counts_in_top_50_genes', 'pct_counts_in_top_100_genes', 'pct_counts_in_top_200_genes', 'pct_counts_in_top_500_genes', 'total_counts_mt', 'log1p_total_counts_mt', 'pct_counts_mt', 'total_counts_ribo', 'log1p_total_counts_ribo', 'pct_counts_ribo', 'n_genes', '_indices', '_scvi_batch', '_scvi_labels', 'B', 'CD8+ T', 'DC', 'NK', 'Resting T', 'Tfh', 'Th2', 'Treg', 'gd T']:
        del adata.obs[k]

    for k in ['mt', 'ribo', 'n_cells_by_counts', 'mean_counts', 'log1p_mean_counts', 'pct_dropout_by_counts', 'total_counts', 'log1p_total_counts', 'SYMBOL', 'MT_gene']:
        del adata.var[k]

    for k in ['MT', 'means_cell_abundance_w_sf', 'q05_cell_abundance_w_sf', 'q95_cell_abundance_w_sf', 'stds_cell_abundance_w_sf']:
        del adata.obsm[k]

    del adata.uns 
    del adata.varm 
    del adata.obsp 

    # %%
    from pathlib import Path
    outdir = Path('/ix/djishnu/shared/djishnu_kor11/training_data_2025')

    # adata.write_h5ad(outdir / f'mouse_lymph{sample}_visiumHD.h5ad')
    adata.write_h5ad(outdir / f'mouse_lymphKO{sample}_visiumHD.h5ad')

    print(f'Done for {sample}')

for sample in [1, 2, 3, 4]:
    annotate(sample)