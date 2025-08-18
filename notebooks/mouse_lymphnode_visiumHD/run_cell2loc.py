import scanpy as sc 
import pandas as pd 
import numpy as np 
import matplotlib.pyplot as plt
import cell2location


sample = 2
tmp_dir = '/ix3/djishnu/alw399/SpaceOracle/data/visiumHD_lymph'
adata = sc.read_h5ad(f'{tmp_dir}/KO_adata_{sample}.h5ad')
# adata = sc.read_h5ad(f'{tmp_dir}/control_adata_{sample}.h5ad')


# rename
adata_vis = adata
adata_vis.X.min(), adata_vis.X.max()

adata_vis.X = np.round(adata_vis.X)

results_folder = './results/lymph_nodes_analysis'

# create paths and names to results folders for reference regression and cell2location models
ref_run_name = f'{results_folder}/reference_signatures_{sample}'
run_name = f'{results_folder}/cell2location_map_{sample}'


adata_vis.var["mt"] = adata_vis.var_names.str.startswith("mt-")
adata_vis.var["ribo"] = adata_vis.var_names.str.startswith(("Rps", "Rpl"))

sc.pp.calculate_qc_metrics(
    adata_vis, qc_vars=["mt", "ribo"], inplace=True, log1p=True
)
sc.pl.violin(
    adata_vis,
    ["n_genes_by_counts", "total_counts", "pct_counts_mt"],
    jitter=0.4,
    multi_panel=True,
)

_ = plt.hist(adata_vis.obs['total_counts'], bins=200)
plt.axvline(x=400, color='red', linestyle='--')

sc.pp.filter_cells(adata_vis, min_genes=50)
# sc.pp.filter_genes(adata_vis, min_cells=15) # we're going to use slideseq genes

print(adata_vis.shape)
adata_vis = adata_vis[adata_vis.obs['total_counts'] > 400]
print(adata_vis.shape)


sc.pl.embedding(
    adata_vis,
    basis='spatial',
    color='total_counts',
    size=10,
    title='Read Depth',
    show=False
)
plt.gca().set_aspect('equal')



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


adata_vis[list(ccr4_prdm1_cells)]


# split adata and run c2l in batches

np.random.seed(42)

batch_size = adata_vis.n_obs // 4
pool = set(adata_vis.obs_names)

parta = np.random.choice(list(pool), size=min(len(pool), batch_size), replace=False)
parta_set = set(parta)

partb = np.random.choice(list(pool - parta_set), size=min(len(pool - parta_set), batch_size), replace=False)
partb_set = set(partb)

partc = np.random.choice(list(pool - parta_set - partb_set), size=min(len(pool - parta_set - partb_set), batch_size), replace=False)
partc_set = set(partc)

partd_set = pool - parta_set - partb_set - partc_set
partd = list(partd_set)

adata_pta = adata_vis[parta].copy()
adata_ptb = adata_vis[partb].copy()
adata_ptc = adata_vis[partc].copy()
adata_ptd = adata_vis[partd].copy()

used = (
    set(adata_ptd.obs_names)
    | set(adata_ptc.obs_names)
    | set(adata_ptb.obs_names)
    | set(adata_pta.obs_names)
)

assert set(pool) == set(used)

adata_pta.shape, adata_ptb.shape, adata_ptc.shape, adata_ptd.shape


adata_vis.var['SYMBOL'] = adata_vis.var_names
# adata_vis.var.set_index('gene_ids', drop=True, inplace=True)

# find mitochondria-encoded (MT) genes
adata_vis.var['MT_gene'] = [gene.startswith('mt-') for gene in adata_vis.var['SYMBOL']]

# remove MT genes for spatial mapping (keeping their counts in the object)
adata_vis.obsm['MT'] = adata_vis[:, adata_vis.var['MT_gene'].values].X.toarray()
adata_vis = adata_vis[:, ~adata_vis.var['MT_gene'].values]

adata_ref= sc.read_h5ad('/ix/djishnu/shared/djishnu_kor11/rctd_outputs/mouse_lymphnode_slideseq/zhongli_ref_202401203_mannually_woDoublet.h5ad')
adata_ref.layers['normalized'] = adata_ref.X.copy()
adata_ref.X = adata_ref.layers['counts']

adata_ref.var['SYMBOL'] = adata_ref.var.index

# delete unnecessary raw slot (to be removed in a future version of the tutorial)
del adata_ref.raw

adata_ref.obs['Subset'] = adata_ref.obs['cell_type']
adata_ref.obs['Method'] = 'GEX'
adata_ref.obs['Sample'] = 'control'

adata_vis.X.min(), adata_vis.X.max()

adata_ref.X.min(), adata_ref.X.max()

from cell2location.utils.filtering import filter_genes
selected = filter_genes(adata_ref, cell_count_cutoff=5, cell_percentage_cutoff2=0.03, nonz_mean_cutoff=1.12)
print(len(selected))
# filter the object
adata_ref = adata_ref[:, selected].copy()


# prepare anndata for the regression model
cell2location.models.RegressionModel.setup_anndata(adata=adata_ref,
    # 10X reaction / sample / batch
    batch_key='Sample',
    # cell type, covariate used for constructing signatures
    labels_key='Subset',
    # multiplicative technical effects (platform, 3' vs 5', donor effect)
    categorical_covariate_keys=['Method']
)

# create the regression model
from cell2location.models import RegressionModel
mod = RegressionModel(adata_ref)

# view anndata_setup as a sanity check
mod.view_anndata_setup()

mod.train(max_epochs=250)
mod.plot_history(20)

# In this section, we export the estimated cell abundance (summary of the posterior distribution).
adata_ref = mod.export_posterior(
    adata_ref, sample_kwargs={'num_samples': 1000, 'batch_size': 2500,
    #  'use_gpu': True
    }
)

# Save model
mod.save(f"{ref_run_name}", overwrite=True)

# Save anndata object with results
adata_file = f"{ref_run_name}/sc.h5ad"
adata_ref.write(adata_file)
adata_file

adata_ref = mod.export_posterior(
    adata_ref, use_quantiles=True,
    # choose quantiles
    # add_to_obsm=["q05","q50", "q95", "q0001"],
    sample_kwargs={'batch_size': 2500,
        # 'use_gpu': True
    }
)

# export estimated expression in each cluster
if 'means_per_cluster_mu_fg' in adata_ref.varm.keys():
    inf_aver = adata_ref.varm['means_per_cluster_mu_fg'][[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
else:
    inf_aver = adata_ref.var[[f'means_per_cluster_mu_fg_{i}'
                                    for i in adata_ref.uns['mod']['factor_names']]].copy()
inf_aver.columns = adata_ref.uns['mod']['factor_names']
inf_aver.iloc[0:5, 0:5]


for part_name, adata_vis in zip(['a', 'b', 'c', 'd'], [adata_pta, adata_ptb, adata_ptc, adata_ptd]):

    # find shared genes and subset both anndata and reference signatures
    intersect = np.intersect1d(adata_vis.var_names, inf_aver.index)
    adata_vis = adata_vis[:, intersect].copy()
    inf_aver = inf_aver.loc[intersect, :].copy()

    # prepare anndata for cell2location model
    cell2location.models.Cell2location.setup_anndata(adata=adata_vis)

    # create and train the model
    mod = cell2location.models.Cell2location(
        adata_vis, cell_state_df=inf_aver,
        # the expected average cell abundance: tissue-dependent
        # hyper-prior which can be estimated from paired histology:
        N_cells_per_location=1,
        # hyperparameter controlling normalisation of
        # within-experiment variation in RNA detection:
        detection_alpha=20
    )
    mod.view_anndata_setup()

    mod.train(max_epochs=18000,
            # train using full data (batch_size=None)
            batch_size=None,
            # use all data points in training because
            # we need to estimate cell abundance at all locations
            train_size=1,
            #   use_gpu=True,
            )

    # plot ELBO loss history during training, removing first 100 epochs from the plot
    mod.plot_history(1000)
    plt.legend(labels=['full data training']);

    # In this section, we export the estimated cell abundance (summary of the posterior distribution).
    adata_vis = mod.export_posterior(
        adata_vis, sample_kwargs={'num_samples': 1000, 'batch_size': mod.adata.n_obs, 
        # 'use_gpu': True
        }
    )

    # Save model
    mod.save(f"{run_name}", overwrite=True)

    # mod = cell2location.models.Cell2location.load(f"{run_name}", adata_vis)

    # add 5% quantile, representing confident cell abundance, 'at least this amount is present',
    # to adata.obs with nice names for plotting
    adata_vis.obs[adata_vis.uns['mod']['factor_names']] = adata_vis.obsm['q05_cell_abundance_w_sf']

    # Save anndata object with results
    adata_file = f"{run_name}/sp_{sample}_part{part_name}.h5ad"
    adata_vis.write(adata_file)
