import SEACells
import matplotlib.pyplot as plt
import seaborn as sns 
import numpy as np

def get_SEACells_assigns(ad, n_SEACells=90, build_kernel_on = 'X_pca', n_waypoint_eigs = 10, show=False):

    model = SEACells.core.SEACells(ad, 
                  build_kernel_on=build_kernel_on, 
                  n_SEACells=n_SEACells, 
                  n_waypoint_eigs=n_waypoint_eigs,
                  convergence_epsilon = 1e-5)

    model.construct_kernel_matrix()
    # M = model.kernel_matrix 

    # Initialize archetypes
    model.initialize_archetypes()
    model.fit(min_iter=10, max_iter=50)

    # Check for convergence 
    if show:
        model.plot_convergence()
        show_soft_assignments(model.A_) 
    
    labels,weights = model.get_soft_assignments()
    return labels, weights


def show_soft_assignments(A, n_top=5):
    '''
    @param A: full assignment matrixs
    '''
    A_transpose = A.T

    plt.figure(figsize=(3,2))
    sns.distplot((A_transpose > 0.1).sum(axis=1), kde=False)
    plt.title(f'Non-trivial (> 0.1) assignments per cell')
    plt.xlabel('# Non-trivial SEACell Assignments')
    plt.ylabel('# Cells')
    plt.show()

    plt.figure(figsize=(3,2))
    b = np.partition(A_transpose, -5)    
    sns.heatmap(np.sort(b[:,-n_top:])[:, ::-1], cmap='viridis', vmin=0)
    plt.title(f'Strength of top {n_top} strongest assignments')
    plt.xlabel('$n^{th}$ strongest assignment')
    plt.show()
