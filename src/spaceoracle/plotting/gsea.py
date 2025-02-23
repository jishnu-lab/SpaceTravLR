import scanpy as sc 
import pandas as pd 
from tqdm import tqdm
import matplotlib.pyplot as plt

def compute_gsea_scores(adata, gsea_modules, layer='imputed_count'):
   
    adata = adata.copy()
    X = adata.layers[layer]
    adata.X = X

    gsea_scores = {}
    for mod_name, mod_dict in tqdm(gsea_modules.items(), total=len(gsea_modules), desc='Computing GSEA scores'):
        gene_list = mod_dict['geneSymbols']
        gene_list = list(set(gene_list) & set(adata.var_names))
        
        if len(gene_list) > 3:
            sc.tl.score_genes(adata, gene_list, score_name=mod_name, use_raw=False)
            gsea_scores[mod_name] = adata.obs[mod_name]
    
    gsea_scores = pd.DataFrame(gsea_scores, columns=gsea_scores.keys()).T
    gsea_scores['score_var'] = gsea_scores.var(axis=1)

    return gsea_scores

def show_gsea_scores(adata, gsea_scores, annot, modules=None, n_show=4, show_spatial=True, savepath=False):

    adata = adata.copy()

    if modules is None:
        modules = list(gsea_scores.head(n_show).index)
       
    for mod in modules:
        adata.obs[mod] = gsea_scores.loc[mod].astype(float)

    plot_params = {
        "color": [annot] + modules,
        "ncols": 5,
        "show": not savepath,
        'cmap': 'viridis'
    }

    if show_spatial:
        plot_params["spot_size"] = 50
        sc.pl.spatial(adata, **plot_params)
    else:
        sc.pp.neighbors(adata)
        sc.tl.umap(adata)
        sc.pl.umap(adata, **plot_params)

    plt.tight_layout()
    if savepath:
        plt.savefig(savepath)

