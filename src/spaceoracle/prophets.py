
import numpy as np
import pandas as pd
from tqdm import tqdm
import json, os 

import matplotlib.pyplot as plt
import seaborn as sns 
import umap

from .models.parallel_estimators import received_ligands
from .oracles import OracleQueue, BaseTravLR
from .beta import Betabase

from .plotting.layout import compare_gex, show_expression_plot, get_grid_layout, show_locations
from .plotting.transitions import estimate_transitions_2D, distance_shift, contour_shift
from .plotting.niche import get_modulator_betas, show_beta_neighborhoods
from .plotting.beta_maps import plot_spatial
from .plotting.location import get_cells_in_radius, show_effect_distance
from .plotting.gsea import compute_gsea_scores, show_gsea_scores

from numba import jit
import enlighten
from concurrent.futures import ThreadPoolExecutor, as_completed



class Prophet(BaseTravLR):
    def __init__(self, adata, models_dir, annot, annot_labels=None, radius=200):

        if annot_labels == None:
            annot_labels = annot
        
        super().__init__(adata, fields_to_keep=[annot, annot_labels])
        
        self.adata = adata.copy()
        self.annot = annot
        self.save_dir = models_dir
        self.annot_labels = annot_labels
        self.radius = radius

        self.queue = OracleQueue(models_dir, all_genes=self.adata.var_names)
        self.ligands = []
        self.genes = list(self.adata.var_names)
        self.trained_genes = []
        self.betas_cache = {}

        self.beta_dict = None
        self.goi = None

        self.gsea_scores = {}
        self.sim_adata = None

        # with open('../../data/GSEA/GSEA_human/h.all.v2024.1.Hs.json', 'r') as f:
        #     self.gsea_modules = json.load(f)
        with open('/ix/djishnu/alw399/SpaceOracle/data/GSEA/m2.all.v2024.1.Mm.json', 'r') as f:
            self.gsea_modules = json.load(f)
        

    def compute_betas(self, subsample=None, float16=False):
        self.beta_dict = self._get_spatial_betas_dict(subsample=subsample, float16=float16)
    
    @staticmethod
    def load_betadata(gene, save_dir):
        return pd.read_parquet(f'{save_dir}/{gene}_betadata.parquet')
    
    def _compute_weighted_ligands(self, gene_mtx):
        gex_df = pd.DataFrame(gene_mtx, index=self.adata.obs_names, columns=self.adata.var_names)

        if len(self.ligands) > 0:
            weighted_ligands = received_ligands(
                xy=self.adata.obsm['spatial'], 
                lig_df=gex_df[self.ligands],
                radius=self.radius
            )
        else:
            weighted_ligands = []
        
        return weighted_ligands
    
    def _get_wbetas_dict(self, betas_dict, weighted_ligands, gene_mtx):

        gex_df = pd.DataFrame(gene_mtx, index=self.adata.obs_names, columns=self.adata.var_names)
        
        with ThreadPoolExecutor() as executor:
            futures = {executor.submit(self._combine_gene_wbetas, weighted_ligands, gex_df, betadata): gene 
                       for gene, betadata in betas_dict.data.items()}
            for future in as_completed(futures):
                gene = futures[future]
                betas_dict.data[gene].wbetas = future.result()

        return betas_dict

    def _combine_gene_wbetas(self, rw_ligands, gex_df, betadata):
        betas_df = betadata.splash(rw_ligands, gex_df)
        return betas_df
        
    def _get_spatial_betas_dict(self, subsample=None, float16=False):
        bdb = Betabase(self.adata, self.save_dir, subsample=subsample, float16=float16)
        self.ligands = list(bdb.ligands_set)
        return bdb
    
    def _perturb_single_cell(self, gex_delta, cell_index, betas_dict):

        genes = self.adata.var_names
        
        gene_gene_matrix = np.zeros((len(genes), len(genes))) # columns are target genes, rows are regulators

        for i, gene in enumerate(genes):
            _beta_out = betas_dict.data.get(gene, None)
            
            if _beta_out is not None:
                r = np.array(_beta_out.modulator_gene_indices)
                gene_gene_matrix[r, i] = _beta_out.wbetas.values[cell_index]

        return gex_delta[cell_index, :].dot(gene_gene_matrix)
    
    
    def _perturb_all_cells(self, gex_delta, betas_dict):
        """
        Vectorized version of _perturb_single_cell.
        For each gene (target), it computes the dot product between the per-cell modulator weights
        and the corresponding gene perturbations.
        
        Returns a matrix of shape (n_obs, n_genes) where each row is the updated perturbation.
        """
        n_obs, n_genes = gex_delta.shape
        result = np.zeros((n_obs, n_genes))
        
        # It may be beneficial to cache modulator indices for each gene if this method is called repeatedly.
        for i, gene in enumerate(self.adata.var_names):
            _beta_out = betas_dict.data.get(gene, None)
            if _beta_out is not None:
                # Precompute modulator indices for the gene
                mod_idx = np.array(_beta_out.modulator_gene_indices)
                
                # Compute the dot product for each cell: multiply the per-cell weights with the corresponding 
                # perturbations and sum over modulator genes.
                result[:, i] = np.sum(_beta_out.wbetas.values * gex_delta[:, mod_idx], axis=1)
        
        return result

    def perturb(self, target, gene_mtx=None, n_propagation=3, gene_expr=0, cells=None, use_optimized=False):

        self.goi = target
        
        for key in ['transition_probabilities', 'grid_points', 'vector_field']:
            self.adata.uns.pop(key, None)

        assert target in self.adata.var_names
        
        if gene_mtx is None: 
            gene_mtx = self.adata.layers['imputed_count']

        if isinstance(gene_mtx, pd.DataFrame):
            gene_mtx = gene_mtx.values

        target_index = self.gene2index[target]  
        simulation_input = gene_mtx.copy()

        # perturb target gene
        if cells is None:
            simulation_input[:, target_index] = gene_expr   
        else:
            simulation_input[cells, target_index] = gene_expr
        
        delta_input = simulation_input - gene_mtx       # get delta X
        delta_simulated = delta_input.copy() 

        if self.beta_dict is None:
            print('Computing beta_dict')
            self.beta_dict = self._get_spatial_betas_dict() # compute betas for all genes for all cells

        weighted_ligands_0 = self._compute_weighted_ligands(gene_mtx)
        weighted_ligands_0 = weighted_ligands_0.reindex(columns=self.adata.var_names, fill_value=0)

        gene_mtx_1 = gene_mtx.copy()

        for n in tqdm(range(n_propagation), desc=f'Perturbing {target}'):

            # weight betas by the gene expression from the previous iteration
            beta_dict = self._get_wbetas_dict(self.beta_dict, weighted_ligands_0, gene_mtx_1)

            # get updated gene expressions
            gene_mtx_1 = gene_mtx + delta_simulated
            weighted_ligands_1 = self._compute_weighted_ligands(gene_mtx_1)
            self.weighted_ligands = weighted_ligands_1

            # update deltas to reflect change in received ligands
            # we consider dy/dwL: we replace delta l with delta wL in  delta_simulated
            weighted_ligands_1 = weighted_ligands_1.reindex(columns=self.adata.var_names, fill_value=0)
            delta_weighted_ligands = weighted_ligands_1.values - weighted_ligands_0.values

            delta_df = pd.DataFrame(delta_simulated, columns=self.adata.var_names, index=self.adata.obs_names)
            delta_ligands = delta_df[self.ligands].reindex(columns=self.adata.var_names, fill_value=0).values
            
            delta_simulated = delta_simulated + delta_weighted_ligands - delta_ligands


            if not use_optimized:
                _simulated = np.array(
                    [self._perturb_single_cell(delta_simulated, i, beta_dict) 
                        for i in tqdm(
                            range(self.adata.n_obs), 
                            desc=f'Running simulation {n+1}/{n_propagation}')])
            else:
                _simulated = self._perturb_all_cells(delta_simulated, beta_dict)
            
            delta_simulated = np.array(_simulated)
            
            # ensure values in delta_simulated match our desired KO / input
            delta_simulated = np.where(delta_input != 0, delta_input, delta_simulated)

            gem_tmp = gene_mtx + delta_simulated
            gem_tmp[gem_tmp<0] = 0
            delta_simulated = gem_tmp - gene_mtx # update delta_simulated in case of negative values

            if target == 'Pax5':
                np.save(f'/ix/djishnu/shared/djishnu_kor11/perturbations/mLDN3-1_v4_{target}/iter_{n}.npy', delta_simulated)

            # save weighted ligand values to weight betas of next iteration
            weighted_ligands_0 = weighted_ligands_1.copy()


        gem_simulated = gene_mtx + delta_simulated
        
        assert gem_simulated.shape == gene_mtx.shape

        # Don't allow simulated to exceed observed values
        imputed_count = gene_mtx
        min_ = imputed_count.min(axis=0)
        max_ = imputed_count.max(axis=0)
        gem_simulated = pd.DataFrame(gem_simulated).clip(lower=min_, upper=max_, axis=1).values

        # Force the gene_expr value for the target gene again
        if cells is None:
            gem_simulated[:, target_index] = gene_expr
        else:
            gem_simulated[cells, target_index] = gene_expr

        self.adata.layers['simulated_count'] = gem_simulated
        self.adata.layers['delta_X'] = gem_simulated - gene_mtx
    
    def perturb_batch(self, target_genes, n_propagation=3, gene_expr=0, cells=None):
        manager = enlighten.get_manager()
        status = manager.status_bar(
            '🚀️ SpaceTravLR',
            color='white_on_black',
            justify=enlighten.Justify.CENTER
        )
        
        for target in tqdm(target_genes, total=len(target_genes)):
            status.update(f'Perturbing {target}')
            status.refresh()
            
            self.perturb(
                target=target, 
                n_propagation=n_propagation, 
                gene_expr=gene_expr, 
                cells=cells, 
                use_optimized=True
            )
    
    def perturb_location(self, coords, goi, n_propagation=3, gene_expr=0, cell_type=[], radius_ko=100, save_dir=None):
        '''perturb a gene in a specific location'''

        cell_idxs = get_cells_in_radius(coords, self.adata, self.annot_labels, radius=radius_ko, cell_type=cell_type)
        self.perturb(goi, n_propagation=n_propagation, gene_expr=gene_expr, cells=cell_idxs)

        # show the effect for each cell type
        top_genes = {}
        for ct in self.adata.obs[self.annot_labels].unique():
            top_gene_labels = self.plot_delta_scores(compare_ct=False, include=[ct], ko_gene=goi, save_dir=save_dir)
            if top_gene_labels is not None:
                top_genes[ct] = top_gene_labels
        
        # show the effect as a wrt distance for each cell type
        show_effect_distance(self.adata, self.annot_labels, top_genes, coords, save_dir=save_dir)

    def show_ct_gex(self, goi):
        df = show_expression_plot(self.adata, goi, self.annot_labels)
        return df

    def evaluate(self, img_dir, perturb_dir, gene_list):
        '''for each cell type, identify the top genes with greatest spatial variation'''
        os.makedirs(img_dir, exist_ok=True)

        for gene in gene_list:

            if gene not in self.adata.var_names:
                print(f'{gene} not found in adata.var_names')
                continue

            simulated_count = os.path.join(perturb_dir, f'{gene}.parquet')
            
            if not os.path.exists(simulated_count):
                self.perturb(gene)
                pd.DataFrame(
                    self.adata.layers['simulated_count'], columns=self.adata.var_names, index=self.adata.obs_names
                    ).to_parquet(f'{perturb_dir}/{gene}.parquet')
            else:
                self.adata.layers['simulated_count'] = pd.read_parquet(simulated_count).values
                self.adata.layers['delta_X'] = self.adata.layers['simulated_count'] - self.adata.layers['imputed_count']
            
            try:
                self.plot_contour_shift(savepath=f'{img_dir}/{gene}/contour.png')
            except:
                print(f'Error in plotting contour for {gene}')
                
            self.plot_delta_scores(save_dir=f'{img_dir}/{gene}')
            self.gene_program_change(savepath=f'{img_dir}/{gene}/delta_gsea.png')

            print(f'finished {gene}')

    
    def plot_contour_shift(self, seed=1334, savepath=False):
        assert self.adata.layers.get('delta_X') is not None

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})
        axs.flatten()
        contour_shift(self.adata, title=f'Cell Identity Shift from {self.goi} KO', annot=self.annot_labels, seed=seed, ax=axs[0])
        
        delta_X_rndm = self.adata.layers['delta_X'].copy()
        permute_rows_nsign(delta_X_rndm)
        fake_simulated_count = self.adata.layers['imputed_count'] + delta_X_rndm
        
        contour_shift(self.adata, title=f'Randomized Effect of {self.goi} KO Shift', annot=self.annot_labels, seed=seed, ax=axs[1], perturbed=fake_simulated_count)
        # axs[1] = distance_shift(self.adata, ax=axs[1], annot=self.annot_labels)
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath)
        plt.show()
    
    def plot_delta_scores(self, n_show=10, compare_ct=True, ct_interest=None, alt_annot=None, include=None, 
                            ko_gene=None, perturbed_cells=None, min_ncells=10, save_dir=False):
        '''
        ct_interest: name of cell type in annot_labels that you want to compare against/ over all others
        alt_annot: group by this annotation instead of cell type
        include: remove all other cell types from consideration
        '''
        if alt_annot is None:
            alt_annot = self.annot_labels
        else:
            assert alt_annot in self.adata.obs.columns, f'{alt_annot} not found in adata.obs'
        
        adata = self.adata.copy()
        
        if ko_gene is not None:
            adata = adata[:, adata.var_names != ko_gene] # remove the gene that was perturbed from the analysis
        if perturbed_cells is not None:
            perturbed_cells = adata.obs_names[np.array(perturbed_cells)]
            adata = adata[:, ~adata.obs.index.isin(perturbed_cells)] # remove the cells that were perturbed from the analysis

        if include is not None:
            adata = self.adata[self.adata.obs[alt_annot].isin(include)]
        
        # Exclude clusters with less than min_ncells 
        cluster_counts = adata.obs[alt_annot].value_counts()
        valid_clusters = cluster_counts[cluster_counts >= min_ncells].index
        adata = adata[adata.obs[alt_annot].isin(valid_clusters)]

        top_gene_labels = distance_shift(adata, alt_annot, n_show=n_show, ct_interest=ct_interest, compare_ct=compare_ct, save_dir=save_dir)
        return top_gene_labels


    def plot_beta_neighborhoods(self, goi=None, use_modulators=False, score_thresh=0.3, savepath=False, seed=1334):
        
        if goi is None:
            goi = self.goi

        if use_modulators:
            # Remove coords and cluster labels
            assert goi in self.beta_dict.data.keys(), f'{goi} does not have modulators'
            betas = self.beta_dict.data[goi].iloc[:, :-4].values
        else:
            betas = self.betas_cache.get(f'betas_{goi}')
            if betas is None:
                self.plot_betas_goi()
                betas = self.betas_cache[f'betas_{goi}']
                
        labels = show_beta_neighborhoods(
            self, goi, betas, 
            annot=self.annot_labels, 
            score_thresh=score_thresh,
            seed=seed,
            savepath=savepath
        )

        self.adata.obs['beta_neighborhood'] = labels
        self.adata.obs['beta_neighborhood'] = self.adata.obs['beta_neighborhood'].astype('category')

    def plot_betas_goi(self, goi=None, save_dir=False, use_simulated=False, clusters=[]):
        '''
        use_simulated: if True, compute rw_ligands from simulated_count, else from imputed_count
        '''
        if goi is None:
            goi = self.goi
        assert goi is not None, 'Specify a gene of interest'

        betas_goi_all = get_modulator_betas(self, goi, save_dir=save_dir, use_simulated=use_simulated, clusters=clusters)
        self.betas_cache[f'betas_{goi}'] = betas_goi_all

    def plot_beta_map(self, regulator, target_gene, clusters=None, save_dir=False):

        if clusters is None:
            clusters = self.adata.obs[self.annot].value_counts().head(3).index

        beta_data = self.beta_dict.data[target_gene]
        beta_data[self.annot_labels] = self.adata.obs[self.annot_labels]

        ax = plot_spatial(
            df=beta_data,
            plot_for=f'beta_{regulator}',
            target_gene=target_gene,
            clusters=clusters,
            annot=self.annot,
            annot_labels=self.annot_labels,
            with_expr=False
        )
        plt.tight_layout()

        if save_dir:
            os.makedirs(save_dir, exist_ok=True)
            savepath = os.path.join(save_dir, f'{target_gene}_beta_{regulator}_map.png')
            plt.savefig(savepath)

    def plot_beta_umap(self, use_modulators=False, seed=1334, n_neighbors=50):

        assert 'beta_neighborhood' in self.adata.obs.columns, f'Run plot_beta_neighborhood() first'

        reducer = umap.UMAP(random_state=seed, n_neighbors=n_neighbors, min_dist=1.0, spread=5.0)
        
        if use_modulators is True:
            X = self.beta_dict.data[self.goi].iloc[:, :-4].values
        else:
            X = self.betas_cache[f'betas_{self.goi}']
        
        umap_coords = reducer.fit_transform(X)

        fig, ax = plt.subplots(figsize=(8, 8))
        sns.scatterplot(
            x=umap_coords[:,0], 
            y=umap_coords[:,1],
            hue=self.adata.obs['beta_neighborhood'].values,
            alpha=0.5,
            s=20,
            ax=ax,
        )
        plt.title(f'Beta UMAP for {self.goi}')

        self.adata.obsm['beta_umap'] = umap_coords

    def show_cluster_gex(self, goi=None, embedding='spatial', annot=None, include_ct=[], show_locs=False):
        if goi is None:
            goi = self.goi

        if annot is None:
            annot = self.annot_labels

        if len(include_ct) > 0:
            adata = self.adata[self.adata.obs[annot].isin(include_ct)]
        else:
            adata = self.adata.copy()
        
        compare_gex(adata, annot=annot, goi=goi, embedding=embedding)
        
        if show_locs:
            show_locations(adata, annot=annot)

    def show_transitions(self, layout_embedding=None, nn_embedding=None, vector_scale=1,
    grid_scale=1, annot=None, n_neighbors=200, n_jobs=1, savepath=False):
            
        fig, axs = plt.subplots(1, 2, figsize=(12, 6))
        axs.flatten()

        if layout_embedding is None:
            layout_embedding = self.adata.obsm['spatial']
        
        if nn_embedding is None:
            nn_embedding = self.adata.obsm['X_draw_graph_fr']
        
        if annot is None:
            annot = self.annot_labels
        
        estimate_transitions_2D(
            adata=self.adata,
            delta_X=self.adata.layers['delta_X'],
            embedding=nn_embedding,
            layout_embedding=layout_embedding,
            annot=annot,
            grid_scale=grid_scale,
            vector_scale=vector_scale,
            n_neighbors=n_neighbors, 
            n_jobs=n_jobs, ax=axs[0]
        )

        delta_X_rndm = self.adata.layers['delta_X'].copy()
        permute_rows_nsign(delta_X_rndm)

        estimate_transitions_2D(
            adata=self.adata,
            delta_X=delta_X_rndm,
            embedding=nn_embedding,
            layout_embedding=layout_embedding,
            annot=annot,
            grid_scale=grid_scale,
            vector_scale=vector_scale,
            n_neighbors=n_neighbors, 
            n_jobs=n_jobs, ax=axs[1]
        )

        fig.suptitle(f"Transition Estimation from {self.goi} KO", fontsize=16)
        axs[0].set_title("Prediction")
        axs[1].set_title("Randomized")
        plt.tight_layout()

        if savepath:
            plt.savefig(savepath)

        plt.show()
    
    def gene_program_change(self, show_spatial=True, savepath=False, keyword='BIOCARTA', n_show=9):
        gsea_scores = compute_gsea_scores(self.adata, self.gsea_modules)
        self.gsea_scores = gsea_scores
        gsea_scores_perturbed = compute_gsea_scores(self.adata, self.gsea_modules, layer='simulated_count')
        self.gsea_scores_ko = gsea_scores_perturbed

        delta_gsea_scores = gsea_scores_perturbed - gsea_scores
        delta_gsea_scores.dropna(inplace=True)

        delta_gsea_scores['abs_mean'] = delta_gsea_scores.iloc[:, :-1].apply(lambda row: np.abs(row.mean()), axis=1)
        delta_gsea_scores.sort_values(by = 'abs_mean', ascending=False, inplace=True)
        
        delta_gsea_scores = delta_gsea_scores.loc[delta_gsea_scores.index.str.contains(keyword)]
        show_gsea_scores(self.adata, delta_gsea_scores, self.annot_labels, n_show=9, show_spatial=True, savepath=savepath)

# Cannibalized from CellOracle
@jit(nopython=True)
def permute_rows_nsign(A: np.ndarray) -> None:
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently.
    """
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])