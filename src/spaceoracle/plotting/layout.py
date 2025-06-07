import matplotlib.pyplot as plt 
import seaborn as sns
import plotly.express as px

from sklearn.neighbors import KDTree

import numpy as np 
import pandas as pd 
import scanpy as sc 


def show_locations(adata, annot):
    data = adata.obsm['spatial']
    df = pd.DataFrame(data, columns=['x', 'y'])
    df['cell_type'] = adata.obs[annot].values

    
    fig = px.scatter(df, x="x", y="y", hover_data=["x", "y"], color='cell_type', opacity=0.7)
    fig.update_traces(
        hovertemplate='%{x:.6f}, %{y:.6f}<extra></extra>', 
        marker=dict(line=dict(width=1, color='DarkSlateGrey'))
    )
    fig.update_layout(
        width=700,
        height=600,
        autosize=False
    )
    fig.show()


def compare_gex(adata, annot, goi, embedding='FR', n_neighbors=15, n_pcs=20, seed=123):

    assert embedding in ['FR', 'PCA', 'UMAP', 'spatial'], f'{embedding} is not a valid embedding choice'
    
    if embedding == 'spatial':
        x = adata.obsm['spatial'][:, 0]
        y = adata.obsm['spatial'][:, 1] * -1

        adata = adata.copy()
        adata.obsm['spatial'] = np.vstack([x, y]).T
        sc.pl.spatial(adata, color=[goi, annot], layer='imputed_count', use_raw=False, cmap='viridis', spot_size=50)

    else:
        sc.tl.pca(adata, svd_solver='arpack')
        sc.pp.neighbors(adata, n_neighbors=n_neighbors, n_pcs=n_pcs)

        if embedding == 'PCA':
            sc.pl.pca(adata, color=[goi, annot], layer='imputed_count', use_raw=False, cmap='viridis')
        
        elif embedding == 'UMAP':
            sc.tl.umap(adata)
            sc.pl.umap(adata, color=[goi, annot], layer='imputed_count', use_raw=False, cmap='viridis')

        elif embedding == 'FR': 

            sc.tl.diffmap(adata)
            sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep='X_diffmap')
            sc.tl.paga(adata, groups=annot)
            sc.pl.paga(adata)

            sc.tl.draw_graph(adata, init_pos='paga', random_state=seed)
            sc.pl.draw_graph(adata, color=[goi, annot], layer="imputed_count", use_raw=False, cmap="viridis", legend_loc='on data')



def plot_quiver(grid_points, vector_field, background=None, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6), dpi=300)

    if background is not None:
        cmap = plt.get_cmap('tab20')
        celltypes = np.unique(background['annot'])
        category_colors = {ct: cmap(i / len(celltypes)) for i, ct in enumerate(celltypes)}
        colors = [category_colors[ct] for ct in background['annot']]

        ax.scatter(background['X'], background['Y'], c=colors, alpha=0.3, s=2)

    magnitudes = np.linalg.norm(vector_field, axis=1)
    indices = magnitudes > 0
    grid_points = grid_points[indices]
    vector_field = vector_field[indices]

    ax.quiver(
        grid_points[:, 0], grid_points[:, 1],   
        vector_field[:, 0], vector_field[:, 1], 
        angles='xy', scale_units='xy', scale=1, 
        headwidth=3, headlength=3, headaxislength=3,
        width=0.002, alpha=0.9
    )

    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('2D Estimated Transition Visualization')
    ax.set_axis_off()

    if ax is not None:
        return ax
    
    plt.show()


def get_grid_layout(layout_embedding, grid_scale=1, create_annot=False, show=False):
    get_grid_points = lambda min_val, max_val: np.linspace(min_val, max_val, 
                            int(np.sqrt((max_val - min_val + 1) * grid_scale))**2)

    grid_x = get_grid_points(np.min(layout_embedding[:, 0]), np.max(layout_embedding[:, 0]))
    grid_y = get_grid_points(np.min(layout_embedding[:, 1]), np.max(layout_embedding[:, 1]))

    if create_annot:

        # Create a grid layout (2D meshgrid)
        grid_xx, grid_yy = np.meshgrid(grid_x, grid_y)
        grid_points = np.vstack([grid_xx.flatten(), grid_yy.flatten()]).T

        # Create a KDTree for fast nearest-neighbor lookup
        tree = KDTree(grid_points)
        sample_labels = tree.query(layout_embedding, k=1)[1].flatten()

        if show:
            scatter = plt.scatter(layout_embedding[:, 0], layout_embedding[:, 1], c=sample_labels, s=10, 
                                  cmap='flag', alpha=0.7, edgecolors='k', linewidth=0.5)
            
            # Draw grid
            for x in grid_x:
                plt.axvline(x=x, color='grey', linestyle='--', alpha=0.7, linewidth=0.5)
            for y in grid_y:
                plt.axhline(y=y, color='grey', linestyle='--', alpha=0.7, linewidth=0.5)

            plt.gca().set_aspect('equal', adjustable='box')

            plt.title('Grid Plot Cell Assignment', fontsize=10)

        return sample_labels

    else:
        return grid_x, grid_y


def show_expression_plot(adata, goi, annot_labels):

    gene_idx = adata.var_names.get_loc(goi)
    expression_data = pd.DataFrame({
        'cluster': adata.obs[annot_labels],
        'expression': adata.layers['imputed_count'][:, gene_idx]
    })

    df = expression_data.groupby('cluster')['expression'].agg(['mean', 'var']).reset_index()
    df.sort_values(by='mean', inplace=True)

    plt.figure(figsize=(8, 4))
    sns.barplot(x='cluster', y='mean', data=df, palette='viridis', order=df['cluster'])
    plt.errorbar(x=df['cluster'], y=df['mean'], yerr=df['var'], fmt='none', c='black', capsize=5)
    plt.title(f'Expression of {goi} across clusters')
    plt.xlabel('Cluster')
    plt.ylabel('Mean Expression')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

    return df