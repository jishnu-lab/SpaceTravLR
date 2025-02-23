
#import packages
import numpy as np
import json 
import scanpy as sc
from collections import OrderedDict
import scipy 
import pandas as pd
import matplotlib.pyplot as plt

#spectra imports 
import Spectra as spc
from Spectra import Spectra_util as spc_tl
from Spectra import K_est as kst
from Spectra import default_gene_sets


obs_key = 'cell_type_annotations'

import json
with open('spectra_outs/annotations.json', 'r') as f:
    annotations = json.load(f)
adata = sc.read_h5ad('spectra_outs/preprocessed.h5ad')


# fit the model (We will run this with only 2 epochs to decrease runtime in this tutorial)
model = spc.est_spectra(adata=adata, 
    gene_set_dictionary=annotations, 
    use_highly_variable=True,
    cell_type_key="cell_type_annotations", 
    use_weights=True,
    lam=0.1, # varies depending on data and gene sets, try between 0.5 and 0.001
    delta=0.001, 
    kappa=None,
    rho=0.001, 
    use_cell_types=True,
    n_top_vals=50,
    label_factors=True, 
    overlap_threshold=0.2,
    clean_gs = True, 
    min_gs_num = 3,
    num_epochs=10000
)

adata.write_h5ad('spectra_outs/mLND3-1_v4_spectra.h5ad')

#You can save the model like this (this way consumes a lot of storage but does not require the model parameters to load)
import pickle
with open('spectra_outs/spectra_model.pickle', 'wb') as f:
    pickle.dump(model, f, pickle.HIGHEST_PROTOCOL)

print('Done!')















