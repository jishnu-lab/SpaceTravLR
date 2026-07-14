import networkx as nx
import numpy as np
import pandas as pd
import os
import scipy as sp
from collections import defaultdict
import matplotlib
import matplotlib.patches as mpatches
from matplotlib.lines import Line2D
import matplotlib.pyplot as plt

'''
Important note: In the perturb function, we sometimes clip the delta of the genes to remain within the previously 
observed values. This is done to prevent the simulation from generating unrealistic gene expression profiles.

Thus, the gradients that we plot may be sometimes greater than the actual effect (since we have clipped the 
actual effect). 
'''


class GraphTracker:
    def __init__(self, adata, gradients, min_delta=1e-5, min_edge=1e-3):
        self.gradients = gradients
        self.adata = adata
        self.min_delta=min_delta
        self.min_edge=min_edge
        self.obs_names = self.adata.obs_names
        self.target_genes = self.adata.var_names
        self.modulator_names = ['beta_' + g for g in self.adata.var_names]

        self.graphs = {layer: self.build_graph(
            obs_names=self.adata.obs_names,
            layer=layer, 
            min_delta=self.min_delta
        ) for layer in gradients.keys()}
        self.G = self.get_total_graph(min_edge_weight=self.min_edge)
        self.G_pt = {}


    def __str__(self):
        lines = [
            f"GraphTracker with {self.G.number_of_nodes()} nodes and {self.G.number_of_edges()} edges",
            f"Layers: {list(self.gradients.keys())}",
            f"Perturbation tracks computed: {len(self.G_pt)}"
        ]
        return "\n".join(lines)

    
    def build_graph(self, obs_names, layer=0, min_delta=1e-5):
        G = nx.DiGraph()
        grad_layer = self.gradients[layer]

        edges = []

        for g in self.target_genes:
            grad_gene = grad_layer.get(g)
            if grad_gene is not None:
                grad_gene = grad_gene.loc[obs_names]
                delta_hop = grad_gene.mean(axis=0)
                delta_hop = delta_hop[delta_hop.abs() > min_delta]
                delta_hop.index = delta_hop.index.str.replace('beta_', '')
                edges.extend([(m, g, delta_hop[m]) for m in delta_hop.index])

        G.add_weighted_edges_from(edges)
        return G

    def get_total_graph(self, min_edge_weight=1e-5):
        G_total = nx.DiGraph()
        all_nodes = sorted(set().union(*(g.nodes() for g in self.graphs.values())))
        
        total_adjacency = sp.sparse.csr_matrix((len(all_nodes), len(all_nodes)))
        
        for layer in self.graphs.values():
            layer_ = layer.copy() 
            layer_.add_nodes_from(all_nodes)
            
            matrix = nx.to_scipy_sparse_array(
                layer_, 
                nodelist=all_nodes, 
                weight='weight'
            )
            total_adjacency += matrix
            
        G_total = nx.from_scipy_sparse_array(total_adjacency, create_using=nx.DiGraph)
        
        mapping = {i: node for i, node in enumerate(all_nodes)}
        G_total = nx.relabel_nodes(G_total, mapping)
        
        # Remove edges with 0 weight
        zero_edges = [(u, v) for u, v, d in G_total.edges(data=True) if abs(d.get('weight', 0)) < min_edge_weight]
        G_total.remove_edges_from(zero_edges)
        
        return G_total

    def get_top_node_mods(self, node, G, n_top=10):
        # 1. Get all incoming edges to this specific node
        # in_edges returns (source, target, data_dict)
        edges = G.in_edges(node, data=True)
        
        # 2. Sort edges by the 'weight' attribute in the data dictionary
        # We use x[2].get('weight', 0) to handle cases where weight might be missing
        sorted_edges = sorted(
            edges, 
            key=lambda x: abs(x[2].get('weight', 0)), 
            reverse=True
        )
        
        # 3. Return a list of (source_node, weight) for your loop to unpack
        return [(src, data.get('weight', 0)) for src, tgt, data in sorted_edges[:n_top]]
    
    def track_perturbation(self, perturb_source, target_gene, n_hops=4, top_n=10):
        G_pt = nx.DiGraph()
        
        # 1. Initialize start/end points
        G_pt.add_node(target_gene, layer=n_hops+1)
        G_pt.add_node(perturb_source, layer=0)

        # We start at the target and work backwards towards the source
        last_layer_nodes = [target_gene]

        # 2. Iterative expansion (Breadth-First Search style)
        for n in range(0, n_hops):
            next_layer_candidates = []
            layer_n = n_hops-n

            for current_node in last_layer_nodes:

                # Get influencers of the current node
                mods = self.get_top_node_mods(current_node, self.G, n_top=top_n)
                
                for mod_gene, weight in mods:

                    if mod_gene not in G_pt:
                        # Add edge and node with metadata
                        # Note: Edge goes mod_gene -> current_node (Influence flow)
                        G_pt.add_node(mod_gene, layer=layer_n)
                        next_layer_candidates.append(mod_gene)
                    
                    G_pt.add_edge(mod_gene, current_node, weight=weight)
                
            if not next_layer_candidates:
                break
            last_layer_nodes = next_layer_candidates

        # 3. Add all edges from all layers to perturbation source
        source_edges = [
            (perturb_source, x, self.G[perturb_source][x]['weight']) for x, d in G_pt.nodes(data=True) 
            if x in self.G[perturb_source]
        ]
        G_pt.add_weighted_edges_from(source_edges)

        # 4. Pruning: Remove "Dead Ends"
        # We only want nodes that are on a path between perturb_source and target_gene
        if perturb_source in G_pt and target_gene in G_pt:
            # Get all nodes that can reach target_gene
            ancestors = nx.ancestors(G_pt, target_gene)
            ancestors.add(target_gene)
            
            # Get all nodes reachable from perturb_source
            descendants = nx.descendants(G_pt, perturb_source)
            descendants.add(perturb_source)
            
            # The intersection contains only nodes on a valid path
            path_nodes = ancestors.intersection(descendants)
            
            # Create a subgraph of just those nodes
            G_pt = G_pt.subgraph(path_nodes).copy()

        return G_pt
    
    def plot_perturbation(self, perturb_source, target_gene, n_hops=3, top_n=10, 
                            ligands=[], receptors=[], figsize=(8,8)):

        G = self.track_perturbation(perturb_source, target_gene, n_hops=n_hops, top_n=top_n)

        self.G_pt[(perturb_source, target_gene)] = G

        color_dict = {
            'ligand': '#8bfcef',
            'receptor': '#69d2ff',
            'TF': 'royalblue',
            'pos_edge': 'salmon',
            'neg_edge': 'lightgreen'
        }

        pos = {}
        nodes_by_layer = defaultdict(list)
        for node, data in G.nodes(data=True):
            nodes_by_layer[data['layer']].append(node)

        for layer, nodes in nodes_by_layer.items():
            n = len(nodes)
            # Spread nodes evenly from y = -0.5 to y = +0.5 for each layer (can tweak spread)
            y_positions = np.linspace(-0.5, 0.5, n) if n > 1 else [0.0]
            for node, y in zip(nodes, y_positions):
                pos[node] = (layer, y)

        fig = plt.figure(figsize=figsize)

        node_colors = []
        for node in G.nodes():
            if node in ligands:
                node_colors.append(color_dict['ligand'])
            elif node in receptors:
                node_colors.append(color_dict['receptor'])
            else: 
                node_colors.append(color_dict['TF'])

        nx.draw_networkx_nodes(
            G, pos, 
            node_size=900, 
            node_color=node_colors, 
            edgecolors='black', 
            linewidths=0.4,
            alpha=0.8,
        )

        edges_data = list(G.edges(data=True))
        edges_weights = []
        edges_colors = []
        for _, _, d in edges_data:
            w = d['weight'] if isinstance(d, dict) and 'weight' in d else 1.0
            edges_weights.append(w)
            if w > 0:
                edges_colors.append(color_dict['pos_edge'])
            else:
                edges_colors.append(color_dict['neg_edge'])
            # edges_colors.append("gray")
    

        max_w = max(abs(float(w)) for w in edges_weights) if edges_weights else 1.0
        widths = [abs(float(w))/(max_w+1e-10)*5+0.5 for w in edges_weights]

        import itertools
        connectionstyles = []
        for idx, (u, v, d) in enumerate(edges_data):
            arc_rad = 0.2 if idx % 2 == 0 else -0.2
            connectionstyles.append(f"arc3,rad={arc_rad}")

        for i, ((u, v, d), color, width, connstyle) in enumerate(zip(edges_data, edges_colors, widths, connectionstyles)):
            nx.draw_networkx_edges(
                G, pos,
                edgelist=[(u, v)],
                width=width,
                edge_color=color,
                arrowstyle='-|>',
                arrowsize=15,
                alpha=0.7,
                connectionstyle=connstyle,
                min_source_margin=10,
                min_target_margin=16
            )

        labels = {node: f"$\mathit{{{node.capitalize()}}}$" for node in G.nodes()}
        nx.draw_networkx_labels(G, pos, labels=labels, font_size=14)

        node_patches = [
            mpatches.Patch(facecolor=color_dict['ligand'], edgecolor='black', label='Ligand'),
            mpatches.Patch(facecolor=color_dict['receptor'], edgecolor='black', label='Receptor'),
            mpatches.Patch(facecolor=color_dict['TF'], edgecolor='black', label='TF'),
        ]

        plt.legend(handles=node_patches,  #+ edge_patches, 
                loc='center left', 
                bbox_to_anchor=(1.02, 0.5),
                frameon=True, fontsize=9, ncol=1, title="Legend", title_fontsize=10)


        plt.title(f"$\mathit{{{perturb_source.capitalize()}}}$ KO → $\mathit{{{target_gene.capitalize()}}}$ expression")
        plt.axis('off')
        return fig
                