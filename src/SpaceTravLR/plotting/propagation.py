"""The layered "how did this perturbation reach that gene" network.

A perturbation summary tells you *that* knocking out a gene moved some readout
gene. This module answers *through what*: it walks backwards from the readout
gene along the strongest tracked gradients, hop by hop, then keeps only what the
perturbed gene can actually reach. What survives is a small directed, layered
graph — ``Nr3c1 -> ... -> Stat3 -> Gata3 -> Il21`` — which is what
:func:`plot_propagation_network` draws. This is the same figure as the
propagation view of the SpaceTravLR webserver.

Input is a :class:`~SpaceTravLR.gradients.GradientTrace` from
``factory.perturb(target=..., track_gradients=True)``.

The walk
--------
1. Start with the readout gene as the only node in the frontier.
2. For each node in the frontier, take its ``n_top`` strongest upstream
   contributions above ``min_abs`` and add them as incoming edges.
3. Repeat ``n_hops`` times. A node keeps the layer of the round it was first
   reached in, which is its shortest backward distance from the readout.
4. Drop everything the perturbed gene cannot reach in ``n_hops`` forward steps.

Step 4 is what makes the picture a *path* rather than a neighbourhood: the
backward walk finds everything that drives the readout, most of which has
nothing to do with the gene that was perturbed.
"""

from collections import defaultdict, deque

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.patches import FancyArrowPatch

__all__ = [
    'PropagationNetwork', 'propagation_network', 'gene_roles',
    'plot_propagation_network', 'ROLE_COLORS',
]

#: Precedence when a gene plays more than one role. A gene that is both a
#: ligand and a receptor is reported as a ligand.
ROLE_ORDER = ('ligand', 'receptor', 'tf')

#: Colours for the roles :func:`gene_roles` assigns — kept in step with the
#: webserver's propagation view.
ROLE_COLORS = {
    'perturbed': '#e15759',
    'readout': '#59a14f',
    'ligand': '#f28e2b',
    'receptor': '#4e79a7',
    'tf': '#b07aa1',
    'other': 'skyblue',
}

ROLE_LABELS = {
    'perturbed': 'Perturbed gene',
    'readout': 'Readout gene',
    'ligand': 'Ligand',
    'receptor': 'Receptor',
    'tf': 'Transcription factor',
    'other': 'Other gene',
}


def gene_roles(betabase, genes=None):
    """What role each gene plays in the trained model.

    Parameters
    ----------
    betabase : Betabase
        A loaded model, i.e. ``factory.beta_dict``.
    genes : sequence of str, optional
        Restrict to these genes.

    Returns
    -------
    dict
        ``{gene: 'ligand' | 'receptor' | 'tf' | 'other'}``. Ligand wins over
        receptor and receptor over TF when a gene is several things — see
        :data:`ROLE_ORDER`.
    """
    members = {
        'ligand': set(betabase.ligands_set) | set(betabase.tfl_ligands_set),
        'receptor': set(betabase.receptors_set),
        'tf': set(betabase.tfs_set),
    }

    if genes is None:
        genes = sorted(set().union(*members.values()))

    roles = {}

    for gene in genes:
        roles[gene] = next(
            (role for role in ROLE_ORDER if gene in members[role]), 'other')

    return roles


class PropagationNetwork:
    """A layered directed graph from a perturbed gene to a readout gene.

    Attributes
    ----------
    source : str
        The perturbed gene. Always layer 0.
    target : str
        The readout gene. Always layer ``n_hops``.
    nodes : dict
        ``{gene: layer}``, in the order the walk discovered them.
    edges : list of tuple
        ``(upstream gene, downstream gene, mean contribution)``. The weight is
        signed: negative means the upstream gene's change *lowered* the
        downstream one.
    n_hops : int
    roles : dict
        Optional ``{gene: role}`` from :func:`gene_roles`.
    label : str
        Which cells the gradients were averaged over.
    emitter_weighted : set of str
        Nodes whose incoming edges were averaged over the cells that emitted the
        ligand rather than over the tracked cells — the diffusible ligands. See
        :mod:`SpaceTravLR.gradients`.
    """

    def __init__(self, source, target, nodes, edges, n_hops, hop=-1, label=None,
                 roles=None, emitter_weighted=()):
        self.source = source
        self.target = target
        self.nodes = dict(nodes)
        self.edges = list(edges)
        self.n_hops = n_hops
        self.hop = hop
        self.label = label
        self.roles = dict(roles) if roles else {}
        self.emitter_weighted = set(emitter_weighted)

    # ------------------------------------------------------------------

    @property
    def n_nodes(self):
        return len(self.nodes)

    @property
    def n_edges(self):
        return len(self.edges)

    def __repr__(self):
        return (
            f'PropagationNetwork({self.source} -> {self.target}, '
            f'{self.n_nodes} nodes, {self.n_edges} edges, '
            f'{self.n_hops} layers)'
        )

    def nodes_by_layer(self):
        """``{layer: [genes]}``, strongest driver first within each layer.

        That is the order the backward walk discovered them in. For drawing,
        use :meth:`layout`, which reorders each column to keep the edges short.
        """
        out = defaultdict(list)

        for gene, layer in self.nodes.items():
            out[layer].append(gene)

        return dict(sorted(out.items()))

    def ordered_layers(self):
        """``{layer: [genes]}`` ordered to keep edges short, for drawing.

        One right-to-left barycentre pass: a node sits at the average height of
        the downstream nodes it feeds, which are already placed. This is the
        standard layered-graph crossing heuristic and it is what makes a
        30-node network legible instead of a hairball.
        """
        downstream = defaultdict(list)

        for gene, node, _ in self.edges:
            downstream[gene].append(node)

        heights = {}
        ordered = {}

        for layer, genes in sorted(self.nodes_by_layer().items(), reverse=True):
            def barycentre(gene):
                placed = [
                    heights[node] for node in downstream.get(gene, ())
                    if node in heights
                ]
                # nothing placed downstream yet: leave it mid-column
                return float(np.mean(placed)) if placed else 0.5

            genes = sorted(genes, key=barycentre)
            ordered[layer] = genes

            for i, gene in enumerate(genes):
                heights[gene] = 0.5 if len(genes) == 1 else i / (len(genes) - 1)

        return dict(sorted(ordered.items()))

    def layout(self, spread=0.5):
        """Positions for a layered drawing: ``{gene: (x, y)}``.

        ``x`` is the layer, ``y`` spreads a layer's nodes evenly over
        ``[-spread, +spread]`` in :meth:`ordered_layers` order.
        """
        positions = {}

        for layer, genes in self.ordered_layers().items():
            ys = (np.linspace(-spread, spread, len(genes))
                  if len(genes) > 1 else [0.0])

            for gene, y in zip(genes, ys):
                positions[gene] = (float(layer), float(y))

        return positions

    @property
    def widest_layer(self):
        """How many nodes are in the most crowded column."""
        return max(len(genes) for genes in self.nodes_by_layer().values())

    def node_frame(self):
        """``gene`` / ``layer`` / ``role`` / ``x`` / ``y``, one row per node."""
        positions = self.layout()

        return pd.DataFrame([
            {
                'gene': gene,
                'layer': layer,
                'role': self.role_of(gene),
                'emitter_weighted': gene in self.emitter_weighted,
                'x': positions[gene][0],
                'y': positions[gene][1],
            }
            for gene, layer in self.nodes.items()
        ])

    def edge_frame(self):
        """``source`` / ``target`` / ``weight``, one row per edge."""
        return pd.DataFrame(self.edges, columns=['source', 'target', 'weight'])

    def role_of(self, gene):
        if gene == self.source:
            return 'perturbed'
        if gene == self.target:
            return 'readout'

        return self.roles.get(gene, 'other')

    def to_dict(self):
        """JSON-serialisable form, with layout positions included."""
        positions = self.layout()

        return {
            'source': self.source,
            'target': self.target,
            'n_hops': self.n_hops,
            'hop': self.hop,
            'cells': self.label,
            'emitter_weighted': sorted(self.emitter_weighted),
            'nodes': [
                {
                    'gene': gene,
                    'layer': int(layer),
                    'role': self.role_of(gene),
                    'emitter_weighted': gene in self.emitter_weighted,
                    'x': positions[gene][0],
                    'y': positions[gene][1],
                }
                for gene, layer in self.nodes.items()
            ],
            'edges': [
                {'source': u, 'target': v, 'weight': float(w)}
                for u, v, w in self.edges
            ],
        }

    def to_networkx(self):
        """The same graph as a ``networkx.DiGraph``, with ``layer`` node attributes."""
        import networkx as nx

        graph = nx.DiGraph()

        for gene, layer in self.nodes.items():
            graph.add_node(gene, layer=layer, role=self.role_of(gene))

        graph.add_weighted_edges_from(self.edges)

        return graph


def propagation_network(trace, source, target, n_top=5, n_hops=5, min_abs=1e-6,
                        hop=-1, roles=None, emitter_weighted=True):
    """Build the layered path from a perturbed gene to a readout gene.

    Parameters
    ----------
    trace : GradientTrace
        From ``factory.perturb(target=source, track_gradients=True)``. Its
        ``track_cells`` decides which cells the edge weights speak for.
    source : str
        The gene that was perturbed — the left-hand side of the picture.
    target : str
        The readout gene to walk back from — the right-hand side.
    n_top : int
        Upstream edges to keep per node. 5 keeps the figure readable; raising
        it finds more paths but fills the middle layers quickly.
    n_hops : int
        Rounds of backward walking, and the forward cutoff from ``source``.
        This is the number of layers, and need not equal the perturbation's
        ``n_propagation``.
    min_abs : float
        Ignore contributions weaker than this.
    hop : int or 'sum'
        Which propagation hop's gradients to read; ``-1`` is the fully
        propagated signal.
    roles : dict, optional
        ``{gene: role}`` from :func:`gene_roles`, for colouring.
    emitter_weighted : bool
        Expand a diffusible ligand's upstream over the cells that emitted what
        the tracked cells received, rather than over the tracked cells.
        Requires the trace to carry those views (``perturb(track_emitters=True)``,
        the default); without them this silently has no effect. Turn it off
        only to reproduce a figure made before the distinction existed —
        averaging a ligand's own expression over cells that did not emit it
        can invert the sign of its incoming edges.

    Returns
    -------
    PropagationNetwork

    Raises
    ------
    ValueError
        If no path from ``source`` to ``target`` exists within ``n_hops``. Try
        more hops, a larger ``n_top``, or a smaller ``min_abs`` — or accept that
        the model routes this perturbation somewhere else.

    Notes
    -----
    Nodes within a layer come out strongest-driver-first, since that is the
    order the walk discovers them in. :meth:`PropagationNetwork.layout` turns
    that into y positions, so the heaviest edges sit at the top of each column.
    """
    if source == target:
        raise ValueError('source and target must be different genes')

    if n_hops < 1:
        raise ValueError('n_hops must be at least 1')

    if trace.upstream(target, hop=hop).empty:
        raise ValueError(
            f'the trace records nothing flowing into {target!r}. Either it has '
            'no fitted model in this dataset — so there is nothing to walk back '
            'along, and it cannot be a readout — or the perturbation never '
            'reached it.')

    layers = {}
    edges = []
    frontier = [target]

    # a diffusible ligand's own expression changed in the cells that emitted
    # it, not in the cells reading out, so its incoming edges are averaged
    # differently
    emitter_genes = trace.emitter_genes if emitter_weighted else set()
    weighted = set()

    for depth in range(n_hops):
        upstream_genes = []

        for node in frontier:
            if node in layers:
                continue

            layers[node] = n_hops - depth

            view = 'emitters' if node in emitter_genes else 'readout'

            if view == 'emitters':
                weighted.add(node)

            contributions = trace.upstream(
                node, hop=hop, min_abs=min_abs, n=n_top, view=view)

            for gene, weight in contributions.items():
                # a self-edge says nothing about how the signal travelled, and
                # would draw as a zero-length arrow
                if gene == node:
                    continue

                edges.append((gene, node, float(weight)))
                upstream_genes.append(gene)

        frontier = upstream_genes

    # -- keep only what the perturbed gene reaches --------------------------

    adjacency = defaultdict(list)

    for gene, node, _ in edges:
        adjacency[gene].append(node)

    distance = {source: 0}
    queue = deque([source])

    while queue:
        node = queue.popleft()

        if distance[node] >= n_hops:
            continue

        for neighbour in adjacency[node]:
            if neighbour not in distance:
                distance[neighbour] = distance[node] + 1
                queue.append(neighbour)

    if target not in distance:
        raise ValueError(
            f'no path from {source!r} to {target!r} within {n_hops} hops at '
            f'n_top={n_top}: {source!r} is not among the strongest {n_top} '
            'drivers of anything on the way. Try more hops, a larger n_top, '
            'or a smaller min_abs.')

    # Nodes discovered in the final backward round were never expanded, so
    # they have no backward layer; fall back to their forward distance from
    # the source, which is bounded by n_hops by construction.
    nodes = {}

    for gene in list(layers) + list(distance):
        if gene in distance and gene not in nodes:
            nodes[gene] = layers.get(gene, distance[gene])

    nodes[source] = 0
    nodes[target] = n_hops

    edges = [
        (gene, node, weight) for gene, node, weight in edges
        if gene in nodes and node in nodes
    ]

    return PropagationNetwork(
        source=source,
        target=target,
        nodes=nodes,
        edges=edges,
        n_hops=n_hops,
        hop=hop,
        label=trace.label,
        roles=roles,
        emitter_weighted=weighted & set(nodes),
    )


def _italic(text):
    return f'$\\mathit{{{text}}}$'


def _colour_by_role(network, genes):
    colours, used = [], {}

    for gene in genes:
        role = network.role_of(gene)
        colour = ROLE_COLORS.get(role, ROLE_COLORS['other'])
        used[role] = colour
        colours.append(colour)

    handles = [
        mpatches.Patch(
            facecolor=colour, edgecolor='black', label=ROLE_LABELS.get(role, role))
        for role, colour in used.items()
    ]

    return colours, handles


def plot_propagation_network(
    network,
    ax=None,
    figsize=None,
    node_size=900,
    node_alpha=0.75,
    signed_edges=True,
    edge_color='gray',
    max_width=3.2,
    arc=0.16,
    font_size=12,
    perturbation='KO',
    title=None,
    legend=True,
    spread=0.5,
):
    """Draw a propagation network as a layered, left-to-right graph.

    This reproduces the propagation view of the SpaceTravLR webserver: nodes
    coloured by role, curved arrows scaled by how much of the downstream
    change came through each edge, and (optionally) coloured red/blue for
    activating/repressing edges.

    Parameters
    ----------
    network : PropagationNetwork
        From :func:`propagation_network`.
    figsize : tuple, optional
        Defaults to a height that grows with the most crowded column, so a
        layer with sixteen genes in it does not draw as sixteen overlapping
        circles.
    signed_edges : bool
        Colour edges by the sign of their weight (red = activating, blue =
        repressing) instead of a uniform grey. Matches the webserver's
        "colour by sign" toggle.
    max_width : float
        Line width of the heaviest edge; the rest scale linearly by
        ``|weight|``.
    arc : float
        Curvature of the edges, alternating in sign so parallel edges
        separate.
    font_size : float
        Label size. A label that overhangs its marker a little is left alone,
        but one that would reach a neighbour is scaled down to fit.
    perturbation : str
        How the perturbation is described in the default title.
    spread : float
        Vertical extent of each column; see :meth:`PropagationNetwork.layout`.

    Returns
    -------
    matplotlib.axes.Axes
    """
    positions = network.layout(spread=spread)
    genes = list(positions)

    if ax is None:
        if figsize is None:
            # the markers are a fixed size in points, so a crowded column only
            # stops overlapping if the figure grows with it
            figsize = (8, max(8, 0.75 * network.widest_layer + 2))

        _, ax = plt.subplots(figsize=figsize, dpi=150)

    colours, handles = _colour_by_role(network, genes)

    # -- edges, behind the nodes -------------------------------------------
    weights = [abs(w) for _, _, w in network.edges]
    heaviest = max(weights) if weights else 1.0

    for i, (source, target, weight) in enumerate(network.edges):
        # alternate the curve so a pair of edges between the same columns does
        # not overlap into one thick line
        rad = arc if i % 2 == 0 else -arc

        if signed_edges:
            colour = '#dc2626' if weight >= 0 else '#2563eb'
        else:
            colour = edge_color

        ax.add_patch(FancyArrowPatch(
            positions[source],
            positions[target],
            connectionstyle=f'arc3,rad={rad}',
            arrowstyle='-|>',
            mutation_scale=15,
            linewidth=abs(weight) / (heaviest + 1e-10) * max_width + 0.6,
            color=colour,
            alpha=0.6,
            # in points, so the arrow stops at the node's edge rather than its
            # centre — the head needs more clearance than the tail
            shrinkA=10,
            shrinkB=16,
            zorder=1,
        ))

    # -- nodes and labels --------------------------------------------------
    xs = np.array([positions[g][0] for g in genes])
    ys = np.array([positions[g][1] for g in genes])

    # a dashed rim marks a node whose incoming edges were averaged over the
    # cells that emitted it rather than over the tracked cells
    emitter = np.array([g in network.emitter_weighted for g in genes])
    styles = np.where(emitter, '--', '-')
    widths = np.where(emitter, 1.4, 0.4)

    ax.scatter(
        xs, ys, s=node_size, c=colours, edgecolors='black', linewidths=widths,
        linestyle=list(styles), alpha=node_alpha, zorder=2,
    )

    # A label a little wider than its marker is normal; only shrink the ones
    # that would run into a neighbour.
    diameter = 2 * np.sqrt(node_size / np.pi)

    for gene in genes:
        x, y = positions[gene]

        # ~0.55 em per character for a proportional italic face
        width = 0.55 * font_size * len(gene)
        size = min(font_size, font_size * 1.4 * diameter / width)

        ax.text(
            x, y, _italic(gene), ha='center', va='center',
            fontsize=size, zorder=3,
        )

    # patches do not autoscale, and the markers need room beyond the data range
    ax.set_xlim(xs.min() - 0.6, xs.max() + 0.6)
    ax.set_ylim(ys.min() - 0.2, ys.max() + 0.2)
    ax.set_axis_off()

    if title is None:
        title = (
            f'{_italic(network.source)} {perturbation} -> '
            f'{_italic(network.target)} expression'
        )

    if title:
        ax.set_title(title)

    if legend and handles and network.emitter_weighted:
        handles = handles + [mpatches.Patch(
            facecolor='none', edgecolor='black', linestyle='--',
            label='upstream over its emitters')]

    if legend and handles:
        if signed_edges:
            handles = handles + [
                mpatches.Patch(facecolor='#dc2626', edgecolor='black',
                               label='activating edge'),
                mpatches.Patch(facecolor='#2563eb', edgecolor='black',
                               label='repressing edge'),
            ]

        ax.legend(
            handles=handles,
            loc='center left',
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=9,
            ncol=1,
            title='Legend',
            title_fontsize=10,
        )

    return ax
