import pandas as pd 
import numpy as np
import matplotlib.pyplot as plt
from pysankey import sankey

from .shift import estimate_transition_probabilities


def get_macrostates(adata, delta_X, embedding, annot, n_neighbors=200, nn_transitions=10):
        transitions = estimate_transition_probabilities(
            adata, 
            delta_X,
            embedding=embedding,
            n_neighbors=n_neighbors,
            annot=annot
        )

        # Get the top nn_transitions transitions
        mask = np.argpartition(transitions, -nn_transitions, axis=1)[:, -nn_transitions:]
        row_indices = np.arange(transitions.shape[0])[:, None]

        transitions = transitions[row_indices, mask]
        idx2ct = {i: ct for i, ct in enumerate(adata.obs[annot])}
        ct_labels = pd.DataFrame(mask).replace(idx2ct)
        
        sankey_data = {'source': [], 'target': [], 'value': []}


        for target_ct in adata.obs[annot].unique():
            ct_mask = np.where(ct_labels == target_ct, 1, 0)
            target_transitions = transitions * ct_mask

            for source_ct in adata.obs[annot].unique():
                ct_transitions = target_transitions[adata.obs[annot] == source_ct]
                sankey_data['source'].append(source_ct)
                sankey_data['target'].append(target_ct)
                sankey_data['value'].append(ct_transitions.sum())

        sankey_df = pd.DataFrame(sankey_data)
        return sankey_df


def plot_pysankey(sankey_df, ax=None):
    if ax is None:
        fig, ax = plt.subplots(1, 1, figsize=(10, 10))
    sankey(
        left=sankey_df['source'], 
        right=sankey_df['target'], 
        leftWeight=sankey_df['value'], 
        rightWeight=sankey_df['value'],
        aspect=20,
        fontsize=12,
        ax=ax
    )
    return ax

