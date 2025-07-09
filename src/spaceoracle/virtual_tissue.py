from functools import cache
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats import pearsonr
from sklearn.preprocessing import StandardScaler
import glob
import enlighten
import numpy as np
import random
import os
from .models.parallel_estimators import create_spatial_features

from .plotting.cartography import Cartography, xy_from_adata
from .gene_factory import GeneFactory
from .beta import BetaFrame

class VirtualTissue:
    
    def __init__(
        self, 
        adata, 
        betadatas_path=None,  
        ko_path=None, 
        ovx_path=None, 
        color_dict=None,
        spf_radius=200,
        annot='cell_type',
        n_props=4
        ):
        
        
        self.adata = adata
        self.betadatas_path = betadatas_path
        self.ko_path = ko_path
        self.ovx_path = ovx_path
        self.annot = annot
        

        self.n_props = n_props
        
        if ovx_path is None:
            self.ovx_path = ko_path
        
        if color_dict is None:
            
            self.color_dict = {
                c: self.random_color() for c in self.adata.obs[self.annot].unique()
            }
        else:
            self.color_dict = color_dict
        
        self.xy = xy_from_adata(self.adata) 
        self.spf_radius = spf_radius
        
        self.spf = create_spatial_features(
            x=adata.obsm['spatial'][:, 0], 
            y=adata.obsm['spatial'][:, 1], 
            celltypes=adata.obs[self.annot], 
            obs_index=adata.obs_names,
            radius = self.spf_radius
        )
        
    def random_color(self):
        return "#{:06x}".format(random.randint(0, 0xFFFFFF))
            
    def load_betadata(self, gene):
        return BetaFrame.from_path(
            f'{self.betadatas_path}/{gene}_betadata.parquet', 
            obs_names=self.adata.obs_names
        )
        
        
    def plot_gene_vs_proximity(
        self, perturb_target, perturbed_df, gene, color_gene, 
        cell_filter, cell_groups, 
        proximity_threshold=150, gene_threshold=0.005, ax=None, mode='ko'):
        
        datadf = self.spf[
            [i+'_within' for i in cell_groups
                ]].sum(1).to_frame().join(self.adata.obs[self.annot]).query(
                f'{self.annot}.isin(["{cell_filter}"])').join(self.xy).join(
            ((perturbed_df-self.adata.to_df(layer='imputed_count'))/self.adata.to_df(layer='imputed_count'))*100
        )
        datadf = datadf[datadf[0] < proximity_threshold]
        
        if ax is None:
            ax = plt.gca()
        
        try:
            corr = pearsonr(datadf[datadf[gene]>gene_threshold][0], datadf[datadf[gene]>gene_threshold][gene]).statistic
            ax.set_title(f"{perturb_target} {mode.upper()} in\n{cell_filter}\nCorrelation: {corr:.4f}")
        except:
            corr = 0
            ax.set_title(f"{perturb_target} {mode.upper()} in\n{cell_filter}")
            
        scatter = ax.scatter(
            datadf[0], 
            datadf[gene], 
            c=datadf[color_gene],
            cmap='rainbow',
        )
        plt.colorbar(scatter, label=f'{color_gene} % change', shrink=0.75, ax=ax, format='%.2f')
        ax.spines['top'].set_visible(False)
        ax.spines['right'].set_visible(False)
        ax.set_ylabel(f'{gene} % change')
        ax.set_xlabel(f'Number of {" & ".join(cell_groups)} cells within {self.spf_radius}um')
        
        return ax, datadf
        
    
    def init_gene_factory(self):

        self.gf = GeneFactory.from_json(
            self.adata, 
            self.betadatas_path + '/run_params.json', 
            override_params={
                'save_dir': self.betadatas_path
            }
        )
        
        
    def init_cartography(self, adata=None, restrict_to=None):
        if adata is None:
            adata = self.adata.copy()
            
        if restrict_to is not None:
            atmp = adata[adata.obs[self.annot].isin(restrict_to)]
        else:
            atmp = adata
        self.chart = Cartography(atmp, self.color_dict)
        
        
    def plot_arrows_pseudotime(self, perturb_target, perturbed_df=None, mode='max', **params):
        if perturbed_df is None:
            perturbed_df = pd.read_parquet(
                f'{self.ovx_path}/{perturb_target}_4n_{mode}x.parquet')
        
        params.setdefault('perturbed_df', perturbed_df)
        params.setdefault('perturb_target', perturb_target)
        params.setdefault('legend_on_loc', True)
        
        grid_points, vector_field, P = self.chart.plot_umap_pseudotime(**params)
        return grid_points, vector_field
    
        
    def plot_arrows(self, perturb_target, perturbed_df=None, mode='max', **params):
        if perturbed_df is None:
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
                
        
        grid_points, vector_field, P = self.chart.plot_umap_quiver(**params)
        
        # grid_points, vector_field, P = self.chart.plot_umap_pseudotime(**params)
        
        
        return grid_points, vector_field

    @cache
    def load_knockout_gex(self, perturb_target):
        ko = pd.read_parquet(
            f"{self.ko_path}/{perturb_target}_4n_0x.parquet"
        )
        
        assert ko[perturb_target].sum() == 0
        
        return ko

    def compute_ko_impact_estimate(
        self, genes, cache_path='', force_recompute=False):
        if os.path.exists(cache_path+'ko_impact_df.csv') and not force_recompute:
            return pd.read_csv(cache_path+'ko_impact_df.csv', index_col=0)
        
        ko_data = []
        files = glob.glob(self.ko_path+f'/{self.n_props}n_0x.parquet')
        
        pbar = enlighten.manager.get_manager().counter(
            total=len(genes),
            desc='Computing KO impact',
            unit='KO',
            auto_refresh=True
        )
        for ko_file in files:
            kotarget = ko_file.split('/')[-1].split('_')[0]

            if kotarget not in genes:
                continue
            
            pbar.desc = f'{kotarget:<15}'
            pbar.refresh()
            
            data = pd.read_parquet(ko_file)
            # data = self.adata.to_df(layer='imputed_count')
            # data[kotarget] = 0
            
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
        out.to_csv(cache_path+'ko_impact_df.csv')
        
        return out
    
    def compute_ko_impact(self, genes, 
        annot='cell_type',
        baseline_only=False,
        ):
        
        # if os.path.exists(cache_path+'ko_impact_df.csv') and not force_recompute:
        #     return pd.read_csv(cache_path+'ko_impact_df.csv', index_col=0)
        
        if isinstance(genes[0], list):
            genes = [g for sublist in genes for g in sublist]
        
        ko_data = []
        if genes is None:  
            files = glob.glob(self.ko_path+'/*_0x.parquet')
        else:
            files = [f"{self.ko_path}/{gene}_4n_0x.parquet" for gene in genes]
        
        pbar = enlighten.manager.get_manager().counter(
            total=len(genes),
            desc='Computing KO impact',
            unit='KO',
            auto_refresh=True
        )
        
        for ko_file in files:
            kotarget = ko_file.split('/')[-1].split('_')[0]

            if kotarget not in genes:
                continue
            
            pbar.desc = f'{kotarget:<15}'
            pbar.refresh()
            
            if baseline_only:
                data = self.adata.to_df(layer='imputed_count')
                data[kotarget] = 0
            else:
                data = pd.read_parquet(ko_file)
            
            data = data.loc[self.adata.obs_names] - self.adata.to_df(layer='imputed_count')
            data = data.join(self.adata.obs[annot]).groupby(annot).mean().abs().mean(axis=1)

            ds = {}
            for k, v in data.sort_values(ascending=False).to_dict().items():
                ds[k] = v

            data = pd.DataFrame.from_dict(ds, orient='index')
            data.columns = [kotarget]
            ko_data.append(data)
            pbar.update(1)
        
        out = pd.concat(ko_data, axis=1)
        # out.to_csv(cache_path+'ko_impact_df.csv')
        
        return out
    
    
    def plot_radar(
        self, 
        genes, 
        impact_df=None, 
        show_for=None, 
        figsize=(20, 6), 
        dpi=300, 
        annot='cell_type', 
        rename=None,
        label_size=20,
        legend_size=12,
        cache_path=None, 
        ):
        
        if isinstance(genes[0], str):
            genes = [genes]
            
        splits = len(genes)
        fig, axs = plt.subplots(1, splits, figsize=figsize, dpi=dpi,
            subplot_kw={'projection': 'polar'})
        
        if splits == 1:
            axs = [axs]
        else:
            axs = axs.flatten()
            
        if impact_df is None:
            impact_df = self.compute_ko_impact(genes=genes, annot=annot)
        
        
        if show_for is not None:
            impact_df = impact_df.loc[show_for]
            
        if rename is not None:
            impact_df.index = impact_df.index.map(lambda x: rename.get(x, x))
        
        ko_concat_norm = pd.DataFrame(
            StandardScaler().fit_transform(impact_df),
            index=impact_df.index, 
            columns=impact_df.columns
        )

        ko_concat_norm = (ko_concat_norm - ko_concat_norm.min().min()) /\
            (ko_concat_norm.max().max() - ko_concat_norm.min().min()) * 100

        for ax, geneset in zip(axs, genes):
            ax.grid(False)
            circles = [0, 25, 50, 75, 100]
            ax.set_rticks(circles)
            ax.set_yticklabels([])
            num_vars = len(ko_concat_norm.index)
            angles = np.linspace(0, 2*np.pi, num_vars, endpoint=False)
            ax.set_rlim(0, 110)
            for circle in circles:
                if circle > 0:  # Skip the center point
                    points = np.array([[circle * np.cos(angle), circle * np.sin(angle)] 
                                     for angle in angles])
                    
                    # Connect the points to form the polygon
                    for i in range(len(points)):
                        j = (i + 1) % len(points)
                        ax.plot([np.arctan2(points[i, 1], points[i, 0]), 
                                np.arctan2(points[j, 1], points[j, 0])],
                               [np.hypot(points[i, 0], points[i, 1]), 
                                np.hypot(points[j, 0], points[j, 1])],
                               color='gray', alpha=0.15, linewidth=0.5)
            
            for angle in angles:
                ax.plot([angle, angle], [0, 110], 
                       color='gray', alpha=0.15, linewidth=0.5)
            
            for i, col in enumerate(geneset):
                values = ko_concat_norm[col].values
                if not np.allclose(values, values[0]):
                    label = rename.get(col, col) if rename is not None else col
                    
                    values_list = values.tolist()
                    values_list += values_list[:1]  # Repeat first value to close polygon
                    
                    angles_plot = np.concatenate((angles, [angles[0]]))  # Complete the polygon

                    ax.plot(angles_plot, values_list, '-', linewidth=1, label=label)
                    ax.fill(angles_plot, values_list, alpha=0.15)

            ax.set_xticks(angles)
            labels = ko_concat_norm.index
            ax.set_xticklabels(
                labels, size=label_size,
                rotation=0  # Keep labels horizontal
            )
            
            ax.tick_params(pad=20)

            ax.spines['polar'].set_visible(False)
            
            legend = ax.legend(bbox_to_anchor=(0.5, -0.15), 
                loc='upper center', ncol=3, frameon=False, fontsize=legend_size)
            if legend:
                for text, line in zip(legend.get_texts(), legend.get_lines()):
                    text.set_color(line.get_color())

        if splits > 1:
            for i in range(1, splits):
                fig.add_artist(plt.Line2D([i/splits, i/splits], [0.1, 0.9], 
                                        transform=fig.transFigure, color='black', 
                                        linestyle='--', linewidth=1, alpha=0.5))
        
        plt.tight_layout()
        
    
class SubsampledTissue(VirtualTissue):
    
    def __init__(
        self, 
        adata, 
        betadatas_paths=None,  
        ko_paths=None, 
        ovx_paths=None, 
        color_dict=None,
        annot='cell_type',
        suffix = '',
        n_props=4
        ):
        
        
        self.adata = adata
        self.betadatas_paths = betadatas_paths
        self.ko_paths = ko_paths
        self.ovx_paths = ovx_paths
        self.annot = annot
        self.suffix = suffix
        self.n_props = n_props

        
        if ovx_paths is None:
            self.ovx_paths = ko_paths
        
        if color_dict is None:
            
            self.color_dict = {
                c: self.random_color() for c in self.adata.obs[self.annot].unique()
            }
        else:
            self.color_dict = color_dict
        
        self.xy = xy_from_adata(self.adata) 


    def compute_ko_impact(self, genes, cache_path='', force_recompute=False):
        if os.path.exists(cache_path+'ko_impact_df.csv') and not force_recompute:
            return pd.read_csv(cache_path+'ko_impact_df.csv', index_col=0)
        
        ko_data = []
        
        pbar = enlighten.manager.get_manager().counter(
            total=len(sum(genes, [])),
            desc='Computing KO impact',
            unit='KO',
            auto_refresh=True
        )
        for kotarget in sum(genes, []):
            pbar.desc = f'{kotarget:<15}'
            pbar.refresh()

            files = [f'{ko_path}/{kotarget}_{self.n_props}n_0x{self.suffix}.parquet' for ko_path in self.ko_paths]
            
            data = pd.concat([
                pd.read_parquet(ko_file) for ko_file in files
            ], axis=0)
            data = data.loc[self.adata.obs.index]
            # data = self.adata.to_df(layer='imputed_count')
            # data[kotarget] = 0
            
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
        out.to_csv(cache_path+'ko_impact_df.csv')
        
        return out
