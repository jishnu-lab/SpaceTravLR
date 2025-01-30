import streamlit as st
import sys
sys.path.append('src')
import torch
import scipy.sparse as sp
import glob
import warnings
import anndata
import matplotlib.pyplot as plt
import scanpy as sc
import numpy as np
import seaborn as sns
import torch
import pandas as pd
import glob
import anndata
import scanpy as sc
import numpy as np
from scipy.ndimage import gaussian_filter

warnings.filterwarnings("ignore", category=FutureWarning)

import spaceoracle
from spaceoracle.tools.network import DayThreeRegulatoryNetwork
from spaceoracle.models.estimators import ViTEstimatorV2, device


@st.cache_data
def load_data():
    adata = sc.read_h5ad(
        '/Users/koush/Desktop/training_data/snrna_human_tonsil.h5ad')
    return adata


adata = load_data()


st.title('🚀️ Welcome to SpaceTravLR')

st.file_uploader('Upload a new spatial transcriptomics dataset', type=['h5ad'])
fig, ax = plt.subplots(figsize=(7, 7))
plt.rcParams['figure.dpi'] = 100

genes = st.multiselect(
    'Select genes to perturb', 
    adata.var_names, 
    default=['FOXO1', 'CXCL13'])

for g in genes:
    min_ = adata.to_df()[g].min()
    max_ = adata.to_df()[g].max()
    mean_ = adata.to_df()[g].mean()
    st.slider(f'Set {g} expression', min_, max_, mean_)

st.button('Perturb! ⚡')

xy = pd.DataFrame(adata.obsm['spatial'], columns=['x', 'y'], index=adata.obs_names)
f, axs = plt.subplots(1, 2, figsize=(11, 5))
axs = axs.flatten()



cell_idx = xy[
    (xy.x > 2000) & (xy.y < 1500)].join(
    adata.obs['cell_type']).query('cell_type=="FDC"').index

cells = adata.obs.reset_index()[adata.obs.reset_index()['NAME'].isin(cell_idx)].index.tolist()

sns.scatterplot(
    x='x', y='y',
    data=xy,
    color='white',
    marker='o',
    edgecolor='black',
    s=10,
    ax=axs[0]
)

sns.scatterplot(
    x='x', y='y',
    data=xy.join(adata.obs['cell_type']).iloc[cells],
    color='red',
    s=20,
    legend=False,
    ax=axs[0]
)


sns.scatterplot(
    x='x', y='y',
    data=xy.join(adata.obs['cell_type']),
    hue='cell_type',
    palette='tab20', 
    linewidth=0.2,
    edgecolor='black',
    s=30,
    ax=axs[1]
)

axs[0].set_axis_off()
axs[1].set_axis_off()

axs[0].set_title('Cells marked for perturbation')


plt.legend(bbox_to_anchor=(0.5, 0), loc='upper center', frameon=False, ncol=3)
ax.set_axis_off()
st.pyplot(f)


df = adata.to_df(layer='imputed_count')[['FOXO1', 'BATF', 'BACH2', 'PRDM1', 'BCL6', 'SATB1', 'ID2', 'PAX5', 
    'CXCR4', 'CD83', 'CD86', 'AICDA', 'BCL2A1', 'BCL2', 'LMO2', 'CXCL13', 
    'CD80', 'TRAF3', 'CCL19', 'CCR7', 'CCL21', 'CD40LG', 'CD40', 'IRF4', 'IRF8', 
    'ITGA5', 'ITGB1', 'ITGAM', 'ITGB2', 'CCR6', 'CD19', 
    'BCL2', 'CD83', 'CD86', 'SDF2', 'CXCR4', 'CXCR5',  'CXCL13', 'CXCL14', 'CXCL12', 'CR2', 'NFKBIZ', 
    'IL6R', 'IL6ST', 'EGR1', 'EGR3', 'EGR2']]


df_with_celltype = df.join(adata.obs['cell_type'])
major_celltypes = df_with_celltype['cell_type'].value_counts().nlargest(4).index.tolist()


df_filtered = df_with_celltype[df_with_celltype['cell_type'].isin(major_celltypes)]
df_long = df_filtered.melt(id_vars=['cell_type'], var_name='Gene', value_name='Expression')

f, axs = plt.subplots(2, 2, figsize=(15, 12))
axs = axs.flatten()

for idx, cell_type in enumerate(major_celltypes):
    cell_data = df_long[df_long['cell_type'] == cell_type]
    
    sns.barplot(
        data=cell_data,
        x='Gene',
        y='Expression',
        ax=axs[idx],
        color='lightblue',
        alpha=0.8,
        ci=None
    )

    axs[idx].set_title(f'{cell_type}')
    axs[idx].tick_params(axis='x', rotation=90)
    axs[idx].set_xlabel('')


plt.tight_layout()
st.pyplot(f)

