import numpy as np 
import pandas as pd
import matplotlib.pyplot as plt
from numba import jit
from glob import glob
import os

import sys
sys.path.append('../../src')
from .plotting.transitions import contour_shift


@jit(nopython=True)
def permute_rows_nsign(A: np.ndarray) -> None:
    """Permute in place the entries and randomly switch the sign for each row of a matrix independently.
    """
    plmi = np.array([+1, -1])
    for i in range(A.shape[0]):
        np.random.shuffle(A[i, :])
        A[i, :] = A[i, :] * np.random.choice(plmi, size=A.shape[1])



class Visionary():
    def __init__(self, adata, perturb_dir, annot_labels='cell_type'):
        self.adata = adata
        self.perturb_dir = perturb_dir
        self.annot_labels = annot_labels
    
    def load_simulated_counts(self, nprops=3, gex=0, gois=None):

        files = glob(f'{self.perturb_dir}/*_{nprops}n_{gex}x.parquet')

        if gois:
            files = [f for f in files if os.path.basename(f).split('_')[0] in gois]

        for f in files:
            goi = os.path.basename(f).split('_')[0]
            self.adata.layers[goi] = pd.read_parquet(f).values

    
    def plot_contour_shift(self, goi, seed=1, savepath=None):

        adata = self.adata.copy()
        adata.layers['simulated_count'] = adata.layers[goi]
        adata.layers['delta_X'] = adata.layers['simulated_count'] - adata.layers['imputed_count']

        fig, axs = plt.subplots(1, 2, figsize=(16, 8), gridspec_kw={'width_ratios': [1, 1]})
        axs.flatten()
        contour_shift(adata, title=f'Cell Identity Shift from {goi} KO', annot=self.annot_labels, seed=seed, ax=axs[0])
        
        delta_X_rndm = adata.layers['delta_X'].copy()
        permute_rows_nsign(delta_X_rndm)
        fake_simulated_count = adata.layers['imputed_count'] + delta_X_rndm
        
        contour_shift(adata, title=f'Randomized Effect of {goi} KO Shift', annot=self.annot_labels, seed=seed, ax=axs[1], perturbed=fake_simulated_count)
        plt.tight_layout()

        if savepath:
            os.makedirs(os.path.dirname(savepath), exist_ok=True)
            plt.savefig(savepath)

        plt.show()
