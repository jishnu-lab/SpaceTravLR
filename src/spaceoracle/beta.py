from functools import partialmethod
import os
import pandas as pd
import numpy as np
import glob
from dataclasses import dataclass
from typing import List, Optional, Tuple
from numba import jit, prange
import numpy as np
from tqdm import tqdm as tqdm_mock
tqdm_mock.__init__ = partialmethod(tqdm_mock.__init__, disable=True)
# import warnings
# warnings.filterwarnings('ignore')

@dataclass
class BetaOutput:
    betas: np.ndarray
    modulator_genes: List[str]
    modulator_gene_indices: List[int]
    ligands: Optional[List[str]] = None
    receptors: Optional[List[str]] = None
    tfl_ligands: Optional[List[str]] = None
    tfl_regulators: Optional[List[str]] = None
    ligand_receptor_pairs: Optional[List[Tuple[str, str]]] = None
    tfl_pairs: Optional[List[Tuple[str, str]]] = None
    wbetas: Optional[Tuple[str, pd.DataFrame]] = None


@jit(nopython=True, parallel=True)
def compute_all_derivatives(tf_vals, lr_betas, lr_ligs, lr_recs, tfl_betas, tfl_ligs, tfl_regs):
    n_samples = tf_vals.shape[0]
    
    # Compute all products in parallel
    rec_derivs = np.zeros((n_samples, lr_betas.shape[1]))
    lig_lr_derivs = np.zeros((n_samples, lr_betas.shape[1]))
    lig_tfl_derivs = np.zeros((n_samples, tfl_betas.shape[1]))
    tf_tfl_derivs = np.zeros((n_samples, tfl_betas.shape[1]))
    
    for i in prange(n_samples):
        # Compute all derivatives in parallel
        rec_derivs[i] = lr_betas[i] * lr_ligs[i]
        lig_lr_derivs[i] = lr_betas[i] * lr_recs[i]
        lig_tfl_derivs[i] = tfl_betas[i] * tfl_regs[i]
        tf_tfl_derivs[i] = tfl_betas[i] * tfl_ligs[i]
    
    return rec_derivs, lig_lr_derivs, lig_tfl_derivs, tf_tfl_derivs
    

class BetaFrame(pd.DataFrame):

    @classmethod
    def from_path(cls, path, cell_index=None, float16=False):
        df = pd.read_parquet(path, engine='pyarrow')
        if float16:
            beta_cols = [col for col in df.columns if col.startswith('beta')]
            df[beta_cols] = df[beta_cols].astype(np.float16)
        if cell_index is not None:
            df = df.loc[cell_index]
        return cls(df)

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

        self.prefix = 'beta_'
        self.tfs = []
        self.lr_pairs = []
        self.tfl_pairs = []

        # to be filled in later
        self.modulator_gene_indices = None
        self.wbetas = None
        
        for col in self.columns:
            if col.startswith(self.prefix):
                modulator = col[len(self.prefix):]
                if '$' in modulator:
                    self.lr_pairs.append(modulator)
                elif '#' in modulator:
                    self.tfl_pairs.append(modulator) 
                else:
                    self.tfs.append(modulator)


        self.ligands, self.receptors = zip(
            *[p.split('$') for p in self.lr_pairs]) if self.lr_pairs else ([], [])
        self.tfl_ligands, self.tfl_regulators = zip(
            *[p.split('#') for p in self.tfl_pairs]) if self.tfl_pairs else ([], [])
        self.ligands = list(self.ligands)
        self.receptors = list(self.receptors)
        self.tfl_ligands = list(self.tfl_ligands)
        self.tfl_regulators = list(self.tfl_regulators)
        self.modulators_genes = [f'beta_{m}' for m in np.unique(
                self.tfs + self.ligands + self.receptors + \
                self.tfl_ligands + self.tfl_regulators)
            ]
        
        self._all_ligands = np.unique(list(self.ligands) + list(self.tfl_ligands))

        # self.df_lr_columns = [f'beta_{r}' for r in self.receptors]+ \
        #     [f'beta_{l}' for l in self.ligands]
        # self.df_tfl_columns = [f'beta_{r}' for r in self.tfl_regulators]+ \
        #     [f'beta_{l}' for l in self.tfl_ligands]
        
        self.tf_columns = [f'beta_{t}' for t in self.tfs]

        self.lr_pairs = [pair.split('$') for pair in self.lr_pairs]
        self.tfl_pairs = [pair.split('#') for pair in self.tfl_pairs]
    

    def splash(self, rw_ligands, gex_df):
        ## wL is the amount of ligand 'received' at each location
        ## assuming ligands and receptors expression are independent, dL/dR = 0
        ## y = b0 + b1*TF1 + b2*wL1R1 + b3*wL1R2
        ## dy/dTF1 = b1
        ## dy/dwL1 = b2[wL1*dR1/dwL1 + R1] + b3[wL1*dR2/dwL1 + R2]
        ##         = b2*R1 + b3*R2
        ## dy/dR1 = b2*[wL1 + R1*dwL1/dR1] = b2*wL1
        
        
        # _df = pd.DataFrame(
        #     np.concatenate([
        #         self[self.tf_columns].to_numpy(),
        #         self[[f'beta_{a}${b}' for a, b in zip(self.ligands, self.receptors)]*2].to_numpy() * \
        #             rw_ligands[self.ligands].join(gex_df[self.receptors]).to_numpy(),
        #         self[[f'beta_{a}#{b}' for a, b in zip(self.tfl_ligands, self.tfl_regulators)]*2].to_numpy() * \
        #             rw_ligands[self.tfl_ligands].join(gex_df[self.tfl_regulators]).to_numpy()
        #     ], axis=1),
        #     index=self.index,
        #     columns=self.tf_columns + self.df_lr_columns + self.df_tfl_columns
        # ).groupby(lambda x: x, axis=1).sum()

        # return _df[self.modulators_genes]
        
                
        lr_betas = self.filter(like='$', axis=1)
        tfl_betas = self.filter(like='#', axis=1)

        rec_derivatives = pd.DataFrame(
            lr_betas.values * rw_ligands[self.ligands].values, 
            index=self.index, 
            columns=self.receptors
        ).astype(float)

        lig_lr_derivatives = pd.DataFrame(
            lr_betas.values * gex_df[self.receptors].values, 
            index=self.index, 
            columns=self.ligands
        ).astype(float)

        lig_tfl_derivatives = pd.DataFrame(
            tfl_betas.values * gex_df[self.tfl_regulators].values, 
            index=self.index, 
            columns=self.tfl_ligands
        ).astype(float)

        tf_derivatives = pd.DataFrame(
            self[self.tf_columns].values,
            index=self.index,
            columns=self.tfs
        ).astype(float)

        tf_tfl_derivatives = pd.DataFrame(
            tfl_betas.values * rw_ligands[self.tfl_ligands].values,
            index=self.index,
            columns=self.tfl_regulators
        ).astype(float)

        _df = pd.concat(
            [
                rec_derivatives, 
                lig_lr_derivatives, 
                lig_tfl_derivatives,
                tf_derivatives,
                tf_tfl_derivatives
            ], axis=1).groupby(level=0, axis=1).sum()

        _df.columns = 'beta_' + _df.columns.astype(str)
        return _df[self.modulators_genes]
    
    
    def splash_fast(self, rw_ligands, gex_df):
        # Extract needed data as numpy arrays for better performance
        tf_values = self[self.tf_columns].values  # Use tf_columns which has the 'beta_' prefix
        lr_ligands = rw_ligands[self.ligands].values
        lr_receptors = gex_df[self.receptors].values
        tfl_ligands = rw_ligands[self.tfl_ligands].values
        tfl_regulators = gex_df[self.tfl_regulators].values
        
        # Get all betas as numpy arrays
        lr_betas = self[[f'beta_{a}${b}' for a, b in zip(self.ligands, self.receptors)]].values
        tfl_betas = self[[f'beta_{a}#{b}' for a, b in zip(self.tfl_ligands, self.tfl_regulators)]].values
        

        # Compute all derivatives at once using JIT
        rec_derivs, lig_lr_derivs, lig_tfl_derivs, tf_tfl_derivs = compute_all_derivatives(
            tf_values, lr_betas, lr_ligands, lr_receptors, tfl_betas, tfl_ligands, tfl_regulators
        )
        
        # Create DataFrames from the computed arrays
        # Create single DataFrame directly instead of list + concat
        columns = self.receptors + self.ligands + self.tfl_ligands + self.tfs + self.tfl_regulators
        values = np.hstack([rec_derivs, lig_lr_derivs, lig_tfl_derivs, tf_values, tf_tfl_derivs])
        _df = pd.DataFrame(values, index=self.index, columns=columns)
        
        # Add prefix and handle duplicates in one step
        _df.columns = 'beta_' + _df.columns.astype(str)
        _df = _df.groupby(_df.columns, axis=1).sum()
        
        return _df[self.modulators_genes]


        
    def _repr_html_(self):
        info = f"BetaFrame with {len(self.modulators_genes)} modulator genes<br>"
        info += f"{len(set(self.tfs))} transcription factors<br>"
        info += f"{len(set(self.ligands))} ligands <br>"
        info += f"{len(set(self.receptors))} receptors <br>"
        info += f"{len(np.unique(self.lr_pairs))} ligand-receptor pairs<br>" 
        info += f"{len(np.unique(self.tfl_pairs))} tfl pairs<br>"
        df_html = super()._repr_html_()
        return f"<div><p>{info}</p>{df_html}</div>"


class Betabase:
    """
    Holds a collection of BetaFrames for each gene.
    """
    def __init__(self, adata, folder, cell_index=None, subsample=None, float16=False):
        assert os.path.exists(folder), f'Folder {folder} does not exist'
        # self.adata = adata
        self.xydf = pd.DataFrame(
            adata.obsm['spatial'], index=adata.obs_names)
        self.folder = folder
        self.gene2index = dict(
            zip(
                adata.var_names, 
                range(len(adata.var_names))
            )
        )
        self.beta_paths = glob.glob(f'{self.folder}/*_betadata.parquet')
        if subsample is not None:
            self.beta_paths = self.beta_paths[:subsample]

        self.data = {}
        self.ligands_set = set()
        self.float16 = float16
        self.load_betas_from_disk(cell_index=cell_index)

    def __len__(self):
        return len(self.data)


    def load_betas_from_disk(self, cell_index):
        from tqdm import tqdm
        for path in tqdm(self.beta_paths):
            gene_name = path.split('/')[-1].split('_')[0]
            self.data[gene_name] = BetaFrame.from_path(path, cell_index=cell_index)
            self.ligands_set.update(self.data[gene_name]._all_ligands)
        
        for gene_name, betadata in self.data.items():
            betadata.modulator_gene_indices = [
                self.gene2index[g.replace('beta_', '')] for g in betadata.modulators_genes
            ]
