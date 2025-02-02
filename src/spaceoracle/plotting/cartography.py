# refer to /scvelo/plotting/velocity_embedding_grid.py
from collections import Counter
import pandas as pd

from matplotlib import pyplot as plt
from scipy.spatial import Delaunay
import numpy as np
import numpy as np
from sklearn.neighbors import NearestNeighbors
from velocyto.estimation import colDeltaCor, colDeltaCorpartial
from tqdm import tqdm
import seaborn as sns
from scipy.spatial import KDTree
import cellrank as cr
from spaceoracle.plotting.shift import estimate_transition_probabilities, project_probabilities
from spaceoracle.plotting.layout import get_grid_layout, plot_quiver


# def alpha_shape(points, alpha, only_outer=True):
#     assert points.shape[0] > 3, "Need at least four points"
#     def add_edge(edges, i, j):
#         """
#         Add an edge between the i-th and j-th points,
#         if not in the list already
#         """
#         if (i, j) in edges or (j, i) in edges:
#             # already added
#             assert (j, i) in edges, "Can't go twice over same directed edge right?"
#             if only_outer:
#                 # if both neighboring triangles are in shape, it's not a boundary edge
#                 edges.remove((j, i))
#             return
#         edges.add((i, j))
#     tri = Delaunay(points)
#     edges = set()
#     # Loop over triangles:
#     # ia, ib, ic = indices of corner points of the triangle
#     for ia, ib, ic in tri.simplices:
#         pa = points[ia]
#         pb = points[ib]
#         pc = points[ic]
#         # Computing radius of triangle circumcircle
#         # www.mathalino.com/reviewer/derivation-of-formulas/derivation-of-formula-for-radius-of-circumcircle
#         a = np.sqrt((pa[0] - pb[0]) ** 2 + (pa[1] - pb[1]) ** 2)
#         b = np.sqrt((pb[0] - pc[0]) ** 2 + (pb[1] - pc[1]) ** 2)
#         c = np.sqrt((pc[0] - pa[0]) ** 2 + (pc[1] - pa[1]) ** 2)
#         s = (a + b + c) / 2.0
#         area = np.sqrt(s * (s - a) * (s - b) * (s - c))

#         circum_r = a * b * c / (4.0 * area)
#         if circum_r < alpha:
#             add_edge(edges, ia, ib)
#             add_edge(edges, ib, ic)
#             add_edge(edges, ic, ia)
    
#     return edges

def xy_from_adata(adata):
    return pd.DataFrame(
        adata.obsm['spatial'], 
        columns=['x', 'y'], 
        index=adata.obs_names
    )
    
class Cartography:
    def __init__(self, adata, color_dict, base_layer='imputed_count'):
        self.adata = adata
        self.xy = xy_from_adata(adata)
        self.base_layer = base_layer
        self.unperturbed_expression = adata.to_df(layer=base_layer)
        self.color_dict = color_dict
                
    @staticmethod
    def compute_perturbation_corr(gene_mtx, delta_X):
        corr = colDeltaCor(
            np.ascontiguousarray(gene_mtx.T), 
            np.ascontiguousarray(delta_X.T), 
        )
        corr = np.nan_to_num(corr, nan=1)
        return corr
    
    
    def get_corr(self, perturb_target):
        assert perturb_target in self.adata.layers
        delta_X = (self.adata.to_df(layer=perturb_target) - self.unperturbed_expression).values
        gene_mtx = self.unperturbed_expression.values
        return self.compute_perturbation_corr(gene_mtx, delta_X)
    

    def compute_transitions(self, corr_mtx, source_ct):

        n_cells = self.adata.shape[0]
        P = np.ones((n_cells, n_cells))
        T = 0.05
        np.fill_diagonal(P, 0)
        P *= np.exp(corr_mtx / T)   
        P /= P.sum(1)[:, None]
        
        transition_df = pd.DataFrame(P[self.adata.obs.banksy_celltypes == source_ct])
        transition_df.columns = self.adata.obs_names
        transition_df.columns.name = source_ct
        return transition_df
    
    @staticmethod
    def assess_transitions(transition_df, base_celltypes, source_ct):
        rx = transition_df.T.join(base_celltypes).groupby('banksy_celltypes').mean()
        rx.columns.name = source_ct
        range_df = pd.DataFrame([rx.min(1), rx.mean(1), rx.max(1)], index=['min', 'mean', 'max']).T
        range_df.columns.name = f'Source Cells: {source_ct}'
        range_df.index.name = 'Transition Target'
        return range_df.sort_values(by='mean', ascending=False)
    
    def get_cellfate(self, transition_df, allowed_fates, thresh=0.0007):
        source_ct = transition_df.columns.name
        assert source_ct in allowed_fates
        transitions = []
        values = []
        base_celltypes = self.adata.obs['banksy_celltypes']
        for ix in tqdm(range(transition_df.shape[0])):
            fate_df = transition_df.iloc[ix].to_frame().join(
                base_celltypes).groupby('banksy_celltypes').mean().loc[allowed_fates]
            
            ct = fate_df.sort_values(ix, ascending=False).iloc[0].to_frame()
            
            self_fate = fate_df.query('banksy_celltypes == @source_ct').values[0][0]
            transition_fate = fate_df.query('banksy_celltypes == @ct.columns[0]').values[0][0]
            
            if transition_fate >= self_fate and transition_fate >= thresh:
                transitions.append(ct.columns[0])
            else:
                transitions.append(source_ct)
            values.append((transition_fate, self_fate))
        
        print(Counter(transitions), np.mean(transition_fate))
        return transitions
    
    def make_celltype_dict(self):
        assert 'transition' in self.adata.obs
        assert 'banksy_celltypes' in self.adata.obs
        
        ct_points_wt = {}
        for ct in self.adata.obs['banksy_celltypes'].unique():
            points = np.asarray(
                self.adata[self.adata.obs['banksy_celltypes'] == ct].obsm['spatial'])
            delta = 30
            points = np.vstack(
                (points +[-delta,delta], points +[-delta,-delta], 
                points +[delta,delta], points +[delta,-delta]))
            ct_points_wt[ct] = points

        ct_points_ko = {}
        for ct in self.adata.obs['transition'].unique():
            points = np.asarray(
                self.adata[self.adata.obs['transition'] == ct].obsm['spatial'])
            delta = 30
            points = np.vstack(
                (points +[-delta,delta], points +[-delta,-delta], 
                points +[delta,delta], points +[delta,-delta]))
            ct_points_ko[ct] = points
            
        return ct_points_wt, ct_points_ko

        
        
    def compute_transition_probabilities(self, delta_X, embedding, n_neighbors=200, remove_null=True, normalize=False):
            
        P = estimate_transition_probabilities(
            self.adata, delta_X, embedding, n_neighbors=n_neighbors, n_jobs=1)
        
        if remove_null:
            P_null = estimate_transition_probabilities(
                self.adata, delta_X * 0, embedding, n_neighbors=n_neighbors, n_jobs=1)
            P = P - P_null
            
        if normalize:
            P = (P - P.min()) / (P.max() - P.min())
            P = P / P.sum(axis=1)[:, np.newaxis]

        return P
    
    
    def plot_umap_quiver(
            self, 
            perturb_target, 
            hue='banksy_celltypes',
            normalize=False, 
            n_neighbors=200, 
            grid_scale=1, 
            vector_scale=1, 
            figsize=(5, 5),
            dpi=180,
            alpha=0.9,
            alt_colors=None
        ):
        assert 'X_umap' in self.adata.obsm
        layout_embedding = self.adata.obsm['X_umap']
        
        delta_X = self.adata.layers[perturb_target] - self.adata.layers['imputed_count']
        
        P = self.compute_transition_probabilities(
            delta_X, 
            layout_embedding, 
            n_neighbors=n_neighbors, 
            remove_null=True
        )
        
        V_simulated = project_probabilities(P, layout_embedding, normalize=normalize)
        
        grid_scale = 10 * grid_scale / np.mean(abs(np.diff(layout_embedding)))
        grid_x, grid_y = get_grid_layout(layout_embedding, grid_scale=grid_scale)
        grid_points = np.array(np.meshgrid(grid_x, grid_y)).T.reshape(-1, 2)
        size_x, size_y = len(grid_x), len(grid_y)
        vector_field = np.zeros((size_x, size_y, 2))
        x_thresh = (grid_x[1] - grid_x[0]) / 2
        y_thresh = (grid_y[1] - grid_y[0]) / 2
        
        get_neighborhood = lambda grid_point, layout_embedding: np.where(
            (np.abs(layout_embedding[:, 0] - grid_point[0]) <= x_thresh) &  
            (np.abs(layout_embedding[:, 1] - grid_point[1]) <= y_thresh)   
        )[0]

        for idx, grid_point in enumerate(grid_points):

            indices = get_neighborhood(grid_point, layout_embedding)
            if len(indices) <= 0:
                continue
            nbr_vector = np.mean(V_simulated[indices], axis=0)
            nbr_vector *= len(indices)       # upweight vectors with lots of cells
                
            grid_idx_x, grid_idx_y = np.unravel_index(idx, (size_x, size_y))
            vector_field[grid_idx_x, grid_idx_y] = nbr_vector

        vector_field = vector_field.reshape(-1, 2)
        
        vector_scale = vector_scale / np.max(vector_field)
        vector_field *= vector_scale
        
            
        f, ax = plt.subplots(figsize=figsize, dpi=dpi)
        
        color_dict = self.color_dict.copy()
        color_dict['GC Dark Zone'] = 'mediumpurple'
        color_dict['GC Intermediate Zone'] = 'mediumpurple'
        color_dict['GC Light Zone'] = 'mediumpurple'

        sns.scatterplot(
            data = pd.DataFrame(
                self.adata.obsm['X_umap'], 
                columns=['x', 'y'], 
                index=self.adata.obs_names).join(self.adata.obs),
            x='x', y='y',
            hue=hue, 
            s=15,
            ax=ax,
            alpha=alpha,
            edgecolor='black',
            linewidth=0.1,
            palette=color_dict,
            legend=False
        )

        plot_quiver(grid_points, vector_field, background=None, ax=ax)

        ax.set_frame_on(False)
        ax.set_xticks([])
        ax.set_yticks([])
        ax.set_xlabel('')
        ax.set_ylabel('')
        ax.set_title('')
        
        if alt_colors is None:
            alt_colors = self.color_dict
            
        if 'alt_labels' in self.adata.obs:
            all_cts = self.adata.obs['alt_labels']
        else:
            all_cts = self.adata.obs[hue]

        for cluster in all_cts.unique():
            cluster_cells = all_cts == cluster
            x = np.mean(self.adata.obsm['X_umap'][cluster_cells, 0])
            y = np.mean(self.adata.obsm['X_umap'][cluster_cells, 1])
            
            ax.text(x, y, cluster, 
                    fontsize=6, 
                    ha='center', 
                    va='center',
                    color='black',
                    bbox=dict(
                        facecolor=alt_colors[cluster],
                        alpha=0.8,
                        edgecolor='black',
                        boxstyle='round'
                    ))
            
        plt.title(f'{perturb_target}')
        
        return ax
    
    
    def get_grids(self, P, projection_params):
        
        self.adata.obsp['_shift'] = P.copy()
        ck = cr.kernels.ConnectivityKernel(self.adata, conn_key='_shift')
        ck.compute_transition_matrix(density_normalize=True)
        
        return ck.plot_projection(**projection_params)
    
    def vector_field_df(self, X_grid, V_grid):
        spatial_coords = self.adata.obsm['spatial']
        grid_tree = KDTree(X_grid)
        dists, idxs = grid_tree.query(spatial_coords, k=4)

        # Convert distances to weights (inverse distance weighting)
        weights = 1 / (dists + 1e-10)  # Add small constant to avoid division by zero
        weights = weights / weights.sum(axis=1, keepdims=True)  # Normalize weights

        # Calculate angles of grid vectors
        grid_angles = np.degrees(np.arctan2(V_grid[:, 1], V_grid[:, 0]))

        # Calculate weighted average angle for each cell
        cell_angles = np.sum(grid_angles[idxs] * weights, axis=1)

        # Create dataframe
        vector_field_df = pd.DataFrame({
            'x': spatial_coords[:, 0],
            'y': spatial_coords[:, 1],
            'angle': cell_angles
        })

        self.adata.obs.index.name = None
        
        vector_field_df.index = self.adata.obs.index
        self.adata.obs.index.name = None
        
        return vector_field_df
        
        
