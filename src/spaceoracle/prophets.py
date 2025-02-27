
import numpy as np
import pandas as pd
from tqdm import tqdm
import commot as ct

from .tools.network import expand_paired_interactions

from .models.parallel_estimators import received_ligands
from .oracles import OracleQueue, BaseTravLR
from .beta import Betabase
from .tools.utils import is_mouse_data
import enlighten

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
        self.lr = self.lr[self.lr.ligand.isin(self.adata.var_names) & (self.lr.receptor.isin(self.adata.var_names))]
        self.lr['radius'] = np.where(
            self.lr['signaling'] == 'Secreted Signaling', 
            self.radius, self.contact_distance
        )

    def compute_betas(self, subsample=None, float16=False):
        self.status.update('Computing betas ...')
        self.status.color = 'black_on_salmon'
        self.status.refresh()

        self.beta_dict = self._get_spatial_betas_dict(
            subsample=subsample, float16=float16)
        
        self.status.update('Computing betas - Done')
        self.status.color = 'black_on_green'
        self.status.refresh()
        
    
    @staticmethod
    def load_betadata(gene, save_dir):
        return pd.read_parquet(f'{save_dir}/{gene}_betadata.parquet')
    
    def _compute_weighted_ligands(self, gene_mtx):
        self.update_status('Computing received ligands', color='black_on_cyan')
        gex_df = pd.DataFrame(gene_mtx, index=self.adata.obs_names, columns=self.adata.var_names)

        if len(self.ligands) > 0:
            weighted_ligands = received_ligands(
                xy=self.adata.obsm['spatial'], 
                ligands_df=gex_df[self.ligands],
                lr_info=self.lr
        )
        else:
            weighted_ligands = []
        
        return weighted_ligands

    
    def update_status(self, msg='', color='black_on_green'):
        self.status.update(msg)
        self.status.color = color
        self.status.refresh()
        
    
    def _get_wbetas_dict(self, betas_dict, weighted_ligands, gene_mtx):

        gex_df = pd.DataFrame(gene_mtx, index=self.adata.obs_names, columns=self.adata.var_names)
        
        for i, (gene, betadata) in enumerate(betas_dict.data.items()):
            betas_dict.data[gene].wbetas = self._combine_gene_wbetas(
                weighted_ligands, gex_df, betadata)
            self.update_status(
                f'[{i:03d}/{len(betas_dict.data):03d}] Ligand interactions', 
                color='black_on_salmon'
            )
            
        self.update_status(f'Ligand interactions - Done')

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
        
        # columns are target genes, rows are regulators
        gene_gene_matrix = np.zeros((len(genes), len(genes))) 

        for i, gene in enumerate(genes):
            _beta_out = betas_dict.data.get(gene, None)
            
            if _beta_out is not None:
                r = np.array(_beta_out.modulator_gene_indices)
                gene_gene_matrix[r, i] = _beta_out.wbetas.values[cell_index]

        return gex_delta[cell_index, :].dot(gene_gene_matrix)
    
    
    def _perturb_all_cells(self, gex_delta, betas_dict):
        n_obs, n_genes = gex_delta.shape
        result = np.zeros((n_obs, n_genes))
        
        self.update_status('Perturbing cells 🐝️', color='black_on_cyan')
        
        for i, gene in enumerate(self.adata.var_names):
            _beta_out = betas_dict.data.get(gene, None)
            if _beta_out is not None:
                mod_idx = np.array(_beta_out.modulator_gene_indices)
                result[:, i] = np.sum(_beta_out.wbetas.values * gex_delta[:, mod_idx], axis=1)
        
        return result

    def perturb(self, target, gene_mtx=None, n_propagation=3, gene_expr=0, cells=None, use_optimized=False, delta_dir=None):

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
            self.beta_dict = self._get_spatial_betas_dict() # compute betas for all genes for all cells

        weighted_ligands_0 = self._compute_weighted_ligands(gene_mtx)
        weighted_ligands_0 = weighted_ligands_0.reindex(columns=self.adata.var_names, fill_value=0)

        gene_mtx_1 = gene_mtx.copy()

        for n in range(n_propagation):
            self.update_status(f'{target} -> {gene_expr} - {n+1}/{n_propagation}', color='black_on_salmon')

            # weight betas by the gene expression from the previous iteration
            beta_dict = self._get_wbetas_dict(self.beta_dict, weighted_ligands_0, gene_mtx_1)

            # get updated gene expressions
            gene_mtx_1 = gene_mtx + delta_simulated
            weighted_ligands_1 = self._compute_weighted_ligands(gene_mtx_1)
            # self.weighted_ligands = weighted_ligands_1

            # update deltas to reflect change in received ligands
            # we consider dy/dwL: we replace delta l with delta wL in  delta_simulated
            weighted_ligands_1 = weighted_ligands_1.reindex(columns=self.adata.var_names, fill_value=0)
            delta_weighted_ligands = weighted_ligands_1.values - weighted_ligands_0.values

            delta_df = pd.DataFrame(
                delta_simulated, columns=self.adata.var_names, index=self.adata.obs_names)
            delta_ligands = delta_df[self.ligands].reindex(
                columns=self.adata.var_names, fill_value=0).values
            
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
            assert not np.isnan(delta_simulated).any(), "NaN values found in delta_simulated"
            
            # ensure values in delta_simulated match our desired KO / input
            delta_simulated = np.where(delta_input != 0, delta_input, delta_simulated)

            # Don't allow simulated to exceed observed values
            gem_tmp = gene_mtx + delta_simulated
            min_ = 0
            max_ = gene_mtx.max(axis=0) * 1.5
            gem_tmp = pd.DataFrame(gem_tmp).clip(lower=min_, upper=max_, axis=1).values

            delta_simulated = gem_tmp - gene_mtx # update delta_simulated in case of negative values

            if delta_dir:
                np.save(f'{delta_dir}/{target}_{n}n_{gene_expr}x.npy', delta_simulated)

            # save weighted ligand values to weight betas of next iteration
            weighted_ligands_0 = weighted_ligands_1.copy()


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
    
    def perturb_batch(self, target_genes, save_to=None, n_propagation=3, gene_expr=0, cells=None):
        
        self.update_status(f'Batch Perturbion mode: {len(target_genes)} genes')
        
        progress_bar = self.manager.counter(
            total=len(target_genes), 
            desc=f'Batch Perturbions', 
            unit='genes',
            color='orange',
            autorefresh=True,
        )
        
        for target in target_genes:
            progress_bar.desc = f'Batch Perturbions - {target}'
            progress_bar.refresh()
            
            self.perturb(
                target=target, 
                n_propagation=n_propagation, 
                gene_expr=gene_expr, 
                cells=cells, 
                use_optimized=True
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
