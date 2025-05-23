import scanpy as sc
import os
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import glob
import enlighten
import numpy as np
import random

from .plotting.cartography import Cartography, xy_from_adata
from .gene_factory import GeneFactory
from .beta import BetaFrame

import jscatter as js

class VirtualTissue:
    
    def __init__(
        self, 
        adata, 
        betadatas_path=None,  
        ko_path=None, 
        ovx_path=None, 
        color_dict=None,
        annot='cell_type'
        ):
        
        
        self.adata = adata
        self.betadatas_path = betadatas_path
        self.ko_path = ko_path
        self.ovx_path = ovx_path
        self.annot = annot
        
        if ovx_path is None:
            self.ovx_path = ko_path
        
        if color_dict is None:
            
            self.color_dict = {
                c: self.random_color() for c in self.adata.obs[self.annot].unique()
            }
        else:
            self.color_dict = color_dict
        
        self.xy = xy_from_adata(self.adata) 
        
    def random_color(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))
            
    def load_betadata(self, gene):
        return BetaFrame.from_path(
            f'{self.betadatas_path}/{gene}_betadata.parquet', 
            obs_names=self.adata.obs_names
        )
        
    
    def init_gene_factory(self):

        self.gf = GeneFactory.from_json(
            self.adata, 
            self.betadatas_path + '/run_params.json', 
            override_params={
                'save_dir': self.betadatas_path
            }
        )
        
        
    def init_cartography(self, adata, restrict_to=None):
        if restrict_to is not None:
            atmp = adata[adata.obs[self.annot].isin(restrict_to)]
        else:
            atmp = adata
        self.chart = Cartography(atmp, self.color_dict)
        
        
    def plot_arrows(self, perturb_target, mode='max', **params):
        perturbed_df = pd.read_parquet(
            f'{self.ovx_path}/{perturb_target}_4n_{mode}x.parquet')
        params.setdefault('perturbed_df', perturbed_df)
        params.setdefault('perturb_target', perturb_target)
        params.setdefault('legend_on_loc', True)
        
        # if params.get('hue', self.annot) is not None:
        #     if not all(x in self.chart.color_dict for x in self.adata.obs[
        #         params.get('hue', self.annot)].unique()):
        #         self.chart.color_dict = {
        #                 c: self.random_color() for c in self.adata.obs[
        #                     params.get('hue', self.annot)].unique()
        #             }
                
        
        _ = self.chart.plot_umap_quiver(**params)

        
        
    def compute_ko_impact(self, force_recompute=False):
        if os.path.exists('ko_impact_df.csv') and not force_recompute:
            return pd.read_csv('ko_impact_df.csv', index_col=0)
        
        ko_data = []
        files = glob.glob(self.ko_path+'/*_0x.parquet')
        
        pbar = enlighten.manager.get_manager().counter(
            total=len(files),
            desc='Computing KO impact',
            unit='KO',
            auto_refresh=True
        )
        for ko_file in files:
            kotarget = ko_file.split('/')[-1].split('_')[0]
            pbar.desc = f'{kotarget:<15}'
            pbar.refresh()
            
            # data = pd.read_parquet(ko_file)
            data = self.adata.to_df(layer='imputed_count')
            data[kotarget] = 0
            
            data = data.loc[self.adata.obs_names] - self.adata.to_df(layer='imputed_count')
            data = data.join(self.adata.obs.cell_type).groupby('cell_type').mean().abs().mean(axis=1)

            ds = {}
            for k, v in data.sort_values(ascending=False).to_dict().items():
                ds[k] = v

            data = pd.DataFrame.from_dict(ds, orient='index')
            data.columns = [kotarget]
            ko_data.append(data)
            pbar.update(1)
        
        out = pd.concat(ko_data, axis=1)
        out.to_csv('ko_impact_df.csv')
        
        return out
    
    
    def plot_radar(self, genes, show_for=None, figsize=(20, 6), dpi=300, rename=None):
        if isinstance(genes[0], str):
            genes = [genes]
            
        splits = len(genes)
        fig, axs = plt.subplots(1, splits, figsize=figsize, dpi=dpi,
            subplot_kw={'projection': 'polar'})
        
        if splits == 1:
            axs = [axs]
        else:
            axs = axs.flatten()
        
        ko_concat = self.compute_ko_impact()
        
        if show_for is not None:
            ko_concat = ko_concat.loc[show_for]
        
        ko_concat_norm = pd.DataFrame(
            StandardScaler().fit_transform(ko_concat), 
            index=ko_concat.index, 
            columns=ko_concat.columns
        )

        for ax, geneset in zip(axs, genes):
            for i, col in enumerate(geneset):
                values = ko_concat_norm[col].values.tolist()
                values += values[:1]  # Repeat first value to close polygon
                
                angles = np.linspace(0, 2*np.pi, len(ko_concat_norm.index), endpoint=False)
                angles = np.concatenate((angles, [angles[0]]))  # Repeat first angle to close polygon

                ax.plot(angles, values, '-', linewidth=1.5, label=col)
                ax.fill(angles, values, alpha=0.1, edgecolor='black', linewidth=0.5, hatch='')

            ax.set_xticks(angles[:-1])
            ax.set_xticklabels(
                ko_concat_norm.index.str.replace(' ', '\n'), size=12,
            )

            ax.set_rlabel_position(0)
            ax.tick_params(pad=15)

            ax.grid(True, alpha=0.1, linestyle='--', color='black')
            ax.set_yticklabels(labels=ax.get_yticks(), size=5)

            ax.spines['polar'].set_visible(False)
            legend = ax.legend(bbox_to_anchor=(0.5, -0.1), 
                loc='upper center', ncol=3, frameon=False, fontsize=14)
            for text, line in zip(legend.get_texts(), legend.get_lines()):
                text.set_color(line.get_color())
            ax.set_rlabel_position(35)
            ax.set_yticklabels([])

        if splits > 1:
            for i in range(1, splits):
                fig.add_artist(plt.Line2D([i/splits, i/splits], [0.1, 0.9], 
                                        transform=fig.transFigure, color='black', 
                                        linestyle='--', linewidth=1, alpha=0.5))
        plt.tight_layout()
        plt.show()
        
        