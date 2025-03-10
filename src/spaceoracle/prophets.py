
from functools import partial
import numpy as np
import pandas as pd
from tqdm import tqdm
import commot as ct
import gc
from .tools.network import expand_paired_interactions
from .models.parallel_estimators import get_filtered_df, received_ligands
from .oracles import OracleQueue, BaseTravLR
from .beta import Betabase
from .tools.utils import is_mouse_data
import enlighten
from pqdm.threads import pqdm
import os


class Prophet(BaseTravLR):
    def __init__(self, adata, models_dir, annot='cell_type_int', radius=100, contact_distance=30):
        
        super().__init__(adata, fields_to_keep=[annot])
        
        
        self.adata = adata.copy()
        self.annot = annot
        self.save_dir = models_dir
        self.radius = radius
        self.contact_distance = contact_distance
        self.species = 'mouse' if is_mouse_data(adata) else 'human'

        self.queue = OracleQueue(models_dir, all_genes=self.adata.var_names)
        self.ligands = []
        self.genes = list(self.adata.var_names)
        self.trained_genes = []
        self.beta_dict = None

        
        self.manager = enlighten.get_manager()
        
        _logo = '🚀️🐭️ SpaceTravLR' if self.species == 'mouse' else '🚀️🙅‍♂️️ SpaceTravLR'
        
        self.status = self.manager.status_bar(
            f'{_logo}: [Ready] | {adata.shape[0]} cells / {len(self.genes)} genes',
            color='black_on_green',
            justify=enlighten.Justify.CENTER,
            auto_refresh=True,
            width=30
        )
        
        self.xy = pd.DataFrame(
            self.adata.obsm['spatial'], 
            index=self.adata.obs_names, 
            columns=['x', 'y']
        )
    
        df_ligrec = ct.pp.ligand_receptor_database(
                database='CellChat', 
                species=self.species, 
                signaling_type=None
            )
            
        df_ligrec.columns = ['ligand', 'receptor', 'pathway', 'signaling']  
        
        self.lr = expand_paired_interactions(df_ligrec)
        self.lr = self.lr[
            self.lr.ligand.isin(self.adata.var_names) & (
                self.lr.receptor.isin(self.adata.var_names))]
        self.lr['radius'] = np.where(
            self.lr['signaling'] == 'Secreted Signaling', 
            self.radius, self.contact_distance
        )
        

    def compute_betas(self, subsample=None, float16=False):
        del self.beta_dict
        gc.collect()
        self.status.update('💾️ Loading betas from disk')
        self.status.color = 'black_on_salmon'
        self.status.refresh()

        self.beta_dict = self._get_spatial_betas_dict(
            subsample=subsample, float16=float16)
        
        self.status.update('Loading betas - Done')
        self.status.color = 'black_on_green'
        self.status.refresh()
        
    
    @staticmethod
    def load_betadata(gene, save_dir):
        return pd.read_parquet(f'{save_dir}/{gene}_betadata.parquet')
    
    def _compute_weighted_ligands(self, gene_mtx, cell_thresholds, genes):
        self.update_status('Computing received ligands', color='black_on_cyan')
        gex_df = pd.DataFrame(
            gene_mtx, 
            index=self.adata.obs_names, 
            columns=self.adata.var_names
        )
        
        if len(self.ligands) > 0:
            weighted_ligands = received_ligands(
                xy=self.adata.obsm['spatial'], 
                ligands_df=get_filtered_df(gex_df, cell_thresholds, genes),
                lr_info=self.lr,
                scale_factor=1
        )
        else:
            weighted_ligands = []
        
        return weighted_ligands

    
    def update_status(self, msg='', color='black_on_green'):
        self.status.update(msg)
        self.status.color = color
        self.status.refresh()
        
        


    def process_gene(self, item, weighted_ligands, weighted_ligands_tfl, filtered_df):
        gene, betadata = item
        return gene, self._combine_gene_wbetas(weighted_ligands, weighted_ligands_tfl, filtered_df, betadata)
            
    
    def _get_wbetas_dict(self, betas_dict, weighted_ligands, weighted_ligands_tfl, gene_mtx, cell_thresholds):

        gex_df = get_filtered_df(       # mask out receptors too
            counts_df=pd.DataFrame(
                gene_mtx, 
                index=self.adata.obs_names, 
                columns=self.adata.var_names
            ),
            cell_thresholds=cell_thresholds,
            genes=self.adata.var_names
        )[self.adata.var_names] 
        
        # out_dict = {}
        self.update_status(f'Computing Ligand interactions', color='black_on_salmon')
        
        process_gene_partial = partial(
            self.process_gene, weighted_ligands=weighted_ligands, weighted_ligands_tfl=weighted_ligands_tfl, filtered_df=gex_df)
        
        results = pqdm(
            betas_dict.data.items(), 
            process_gene_partial, 
            n_jobs=8, 
            tqdm_class=tqdm
        )
        
        # for i, (gene, betadata) in enumerate(betas_dict.data.items()):
        #     # betas_dict.data[gene].wbetas = self._combine_gene_wbetas(
        #     #     weighted_ligands, gex_df, betadata)
        #     out_dict[gene] = self._combine_gene_wbetas(
        #         weighted_ligands, gex_df, betadata)
            
        #     self.update_status(
        #         f'{self.iter}/{self.max_iter} | [{i/len(betas_dict.data)*100:5.1f}%] Ligand interactions splash', 
        #         color='black_on_salmon'
        #     )
            
        self.update_status(f'Ligand interactions - Done')

        return dict(results)

    def _combine_gene_wbetas(self, rw_ligands, rw_ligands_tfl, filtered_df, betadata):
        # betas_df = betadata.splash_fast(rw_ligands, rw_ligands_tfl, gex_df) ## this works but doesn't seem faster
        betas_df = betadata.splash(
            rw_ligands, 
            rw_ligands_tfl, 
            filtered_df
        )
        
        return betas_df
        
    def _get_spatial_betas_dict(self, subsample=None, float16=False):
        bdb = Betabase(self.adata, self.save_dir, subsample=subsample, float16=float16)
        self.ligands = list(bdb.ligands_set)
        self.tfl_ligands = list(bdb.tfl_ligands_set)
        return bdb
    
    def _perturb_single_cell(self, gex_delta, cell_index, betas_dict):

        genes = self.adata.var_names
        
        # columns are target genes, rows are regulators
        gene_gene_matrix = np.zeros((len(genes), len(genes))) 

        for i, gene in enumerate(genes):
            _beta_out = betas_dict.get(gene, None)
            
            if _beta_out is not None:
                # r = np.array(_beta_out.modulator_gene_indices)
                r = np.array(self.beta_dict.data[gene].modulator_gene_indices)
                gene_gene_matrix[r, i] = _beta_out.values[cell_index]

        return gex_delta[cell_index, :].dot(gene_gene_matrix)
    
    
    def _perturb_all_cells(self, gex_delta, betas_dict):
        n_obs, n_genes = gex_delta.shape
        result = np.zeros((n_obs, n_genes))
        n_vars = len(self.adata.var_names)
        
        
        for i, gene in enumerate(self.adata.var_names):
            self.update_status(
                f'[{self.iter}/{self.max_iter}] | {i+1}/{n_vars} | Perturbing cells 🐝️', 
                color='black_on_cyan'
            )
            
            _beta_out = betas_dict.get(gene, None)
            if _beta_out is not None:

                # mod_idx = np.array(_beta_out.modulator_gene_indices)
                mod_idx = np.array(self.beta_dict.data[gene].modulator_gene_indices)
                
                result[:, i] = np.sum(_beta_out.values * gex_delta[:, mod_idx], axis=1)
        return result

    def perturb(self, target, gene_mtx=None, n_propagation=2, gene_expr=0, 
                cells=None, use_optimized=True, delta_dir=None, retain_propagation=False):

        assert target in self.adata.var_names
        
        if retain_propagation:
            propagations = []
        
        if gene_mtx is None: 
            gene_mtx = self.adata.layers['imputed_count'].copy()

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
            self.beta_dict = self._get_spatial_betas_dict() # compute betas for all genes for all cells

        # get LR specific filtered gex contributions
        cell_thresholds = self.adata.uns.get('cell_thresholds')
        if cell_thresholds is not None:
            cell_thresholds = cell_thresholds.reindex(              # Only commot LR values should be filtered
                index=self.adata.obs_names, columns=self.adata.var_names, fill_value=1)
            self.adata.uns['cell_thresholds'] = cell_thresholds
        else:
            print('warning: cell_thresholds not found in adata.uns')

        w_ligands_0 = self.adata.uns.get('received_ligands')
        w_tfligands_0 = self.adata.uns.get('received_ligandds_tfl')
        if w_ligands_0 is None or w_tfligands_0 is None:
            w_ligands_0 = self._compute_weighted_ligands(gene_mtx, cell_thresholds, genes=self.ligands)
            w_tfligands_0 = self._compute_weighted_ligands(gene_mtx, cell_thresholds=None, genes=self.tfl_ligands)
        
        weighted_ligands_0 = pd.concat(
                [w_ligands_0, w_tfligands_0], axis=1
            ).groupby(level=0, axis=1).max().reindex(
                index=self.adata.obs_names, columns=self.adata.var_names, fill_value=0)

        gene_mtx_1 = gene_mtx.copy()
        
        self.iter = 0
        self.max_iter = n_propagation
        min_ = 0
        max_ = gene_mtx.max(axis=0)
        
        ## refer: src/celloracle/trajectory/oracle_GRN.py

        for n in range(n_propagation):
            self.iter+=1
            self.update_status(f'{target} -> {gene_expr} - {n+1}/{n_propagation}', color='black_on_salmon')

            # weight betas by the gene expression from the previous iteration
            beta_dict = self._get_wbetas_dict(
                self.beta_dict, w_ligands_0, w_tfligands_0, gene_mtx_1, cell_thresholds)

            # get updated gene expressions
            gene_mtx_1 = gene_mtx + delta_simulated
            w_ligands_1 = self._compute_weighted_ligands(gene_mtx_1, cell_thresholds, genes=self.ligands)
            w_tfligands_1 = self._compute_weighted_ligands(gene_mtx_1, cell_thresholds=None, genes=self.tfl_ligands)

            # update deltas to reflect change in received ligands
            # we consider dy/dwL: we replace delta l with delta wL in  delta_simulated
            weighted_ligands_1 = pd.concat(
                    [w_ligands_1, w_tfligands_1], axis=1
                ).groupby(level=0, axis=1).max().reindex(
                    index=self.adata.obs_names, columns=self.adata.var_names, fill_value=0)
            delta_weighted_ligands = weighted_ligands_1.values - weighted_ligands_0.values

            delta_df = pd.DataFrame(
                delta_simulated, columns=self.adata.var_names, index=self.adata.obs_names)
            delta_ligands = pd.concat(
                    [delta_df[self.ligands], delta_df[self.tfl_ligands]], axis=1
                ).groupby(level=0, axis=1).max().reindex(
                    index=self.adata.obs_names, columns=self.adata.var_names, fill_value=0).values
            
            delta_simulated = delta_simulated + delta_weighted_ligands - delta_ligands
            _simulated = self._perturb_all_cells(delta_simulated, beta_dict)
            # delta_simulated = delta_simulated + np.array(_simulated)
            
            assert not np.isnan(_simulated).any(), "NaN values found in delta_simulated"
            
            # ensure values in delta_simulated match our desired KO / input
            delta_simulated = np.where(delta_input != 0, delta_input, delta_simulated)
            print(pd.DataFrame(delta_simulated, index=self.adata.obs_names, columns=self.adata.var_names))

            # Don't allow simulated to exceed observed values
            gem_tmp = gene_mtx + delta_simulated
            gem_tmp = pd.DataFrame(gem_tmp).clip(lower=min_, upper=max_, axis=1).values

            delta_simulated = gem_tmp - gene_mtx # update delta_simulated in case of negative values
            
            if delta_dir:
                os.makedirs(delta_dir, exist_ok=True)
                np.save(f'{delta_dir}/{target}_{n}n_{gene_expr}x.npy', delta_simulated)

            if retain_propagation:
                propagations.append(gene_mtx + delta_simulated)
            
            del beta_dict
            gc.collect()
        
        gem_simulated = gene_mtx + delta_simulated
        assert gem_simulated.shape == gene_mtx.shape

        # Force the gene_expr value for the target gene again
        if cells is None:
            gem_simulated[:, target_index] = gene_expr
        else:
            gem_simulated[cells, target_index] = gene_expr

        # self.adata.layers['simulated_count'] = gem_simulated
        # self.adata.layers['delta_X'] = gem_simulated - gene_mtx
        self.adata.layers[f'{target}_{n_propagation}n_{gene_expr}x'] = gem_simulated
        
        # print(f'Layer added: {target}_{n_propagation}n_{gene_expr}x')
        
        self.update_status(f'{target} -> {gene_expr} - {n_propagation}/{n_propagation} - Done')
        
        if retain_propagation:
            return propagations
        
    
    def perturb_batch(self, target_genes, save_to=None, n_propagation=3, gene_expr=0, cells=None, delta_dir=None):
        
        self.update_status(f'Batch Perturbion mode: {len(target_genes)} genes')
        
        progress_bar = self.manager.counter(
            total=len(target_genes), 
            desc=f'Batch Perturbions', 
            unit='genes',
            color='orange',
            autorefresh=True,
        )

        os.makedirs(save_to, exist_ok=True)
        
        for target in target_genes:
            progress_bar.desc = f'Batch Perturbions - {target}'
            progress_bar.refresh()
            
            self.perturb(
                target=target, 
                n_propagation=n_propagation, 
                gene_expr=gene_expr, 
                cells=cells, 
                use_optimized=True,
                delta_dir=delta_dir
            )
                     
            progress_bar.update()
            
            
            file_name = f'{target}_{n_propagation}n_{gene_expr}x'
            
            if save_to is not None:
                self.adata.to_df(
                    layer = file_name).to_parquet(
                        f'{save_to}/{file_name}.parquet')
                    
                del self.adata.layers[file_name] # save memory
                
        self.update_status('Batch Perturbions: Done')
        progress_bar.close()
