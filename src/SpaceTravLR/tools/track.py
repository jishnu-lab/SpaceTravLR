import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import os

def _strip_beta_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Return a DataFrame whose columns are gene symbols (beta_ prefix removed). Never mutates ``df``."""
    if df.columns.empty:
        return df
    new_cols = [c[5:] if isinstance(c, str) and c.startswith("beta_") else str(c) for c in df.columns]
    return df.set_axis(new_cols, axis=1, copy=False)


class GraphTracker:
    def __init__(self):
        self.G = nx.DiGraph()
        self.gradients = None  # Store gradients if you want
        self.modulator_list = None

    @staticmethod
    def get_mod_effect(gradient_layer, modulator_list):
        tot_effect = {}
        for target_gene, betadata in gradient_layer.items():
            if not isinstance(betadata, pd.DataFrame) or betadata.empty:
                tot_effect[target_gene] = pd.Series(0.0, index=modulator_list)
                continue
            sub = _strip_beta_columns(betadata)
            effect = sub.reindex(columns=modulator_list, fill_value=0).mean(axis=0)
            tot_effect[target_gene] = effect
        return tot_effect

    @staticmethod
    def get_path_effect(G, paths):
        """
        Per-path score = weight of the **last edge only** (penultimate node → final node).

        For paths ``source → … → target``, this is the directed edge
        ``path[-2] → path[-1]`` (e.g. second-to-last gene → ``Tap1``). All earlier
        hops are ignored for ranking. Paths shorter than two nodes score 0.
        """
        n = len(paths)
        if n == 0:
            return []
        out = np.zeros(n, dtype=np.float64)
        try:
            succ_map = G._succ
        except AttributeError:
            succ_map = None
        for i, path in enumerate(paths):
            if len(path) < 2:
                continue
            u, v = path[-2], path[-1]
            if succ_map is not None:
                dat = succ_map.get(u, {}).get(v)
            else:
                dat = G.get_edge_data(u, v)
            if dat:
                w = dat.get("weight", 0.0)
                out[i] = 0.0 if w is None else float(w)
        return out

    @staticmethod
    def get_top_paths(G, source, target, cutoff, n_paths=10):
        """
        All simple ``source → target`` paths within ``cutoff`` edges, ranked by
        largest **|final-edge weight|** (absolute value of :meth:`get_path_effect`).
        """
        paths = list(nx.all_simple_paths(G, source=source, target=target, cutoff=cutoff))
        edge_values = GraphTracker.get_path_effect(G, paths)
        m = len(paths)
        if m == 0:
            return []
        k = min(n_paths, m)
        ev = np.asarray(edge_values, dtype=np.float64)
        ev = abs(ev)
        if k == m:
            top_idx = np.argsort(ev)[::-1]
        else:
            idx = np.argpartition(-ev, k - 1)[:k]
            top_idx = idx[np.argsort(ev[idx])[::-1]]
        return [paths[i] for i in top_idx]

    @staticmethod
    def _edge_score(G, u, v, use_abs: bool) -> float:
        d = G.get_edge_data(u, v)
        if not d:
            return 0.0
        w = d.get("weight", 0.0)
        if w is None:
            w = 0.0
        w = float(w)
        return abs(w) if use_abs else w

    @staticmethod
    def get_iterative_distinct_paths(
        G,
        source,
        target,
        cutoff,
        n_paths: int = 10,
        greedy_use_abs: bool = True,
    ):
        """
        Build up to ``n_paths`` **simple** paths ``source → … → target`` such that
        each path uses a **different** penultimate node ``U`` (edge ``U → target``).

        Penultimate candidates are ordered by strongest ``U → target`` (see
        ``greedy_use_abs``). For each ``U``, walk **backward** from ``U`` toward
        ``source``, each step choosing the in-neighbor with the strongest edge into
        the current node (same ``use_abs`` rule). If that trace does not reach
        ``source`` within ``cutoff`` edges, falls back to an unweighted
        ``nx.shortest_path(G, source, U)`` when ``sp + [target]`` is simple and
        within the edge budget.

        ``cutoff`` matches NetworkX path length: max number of **edges** in the path.
        """
        if source not in G or target not in G:
            return []

        # (penultimate, score on U->target)
        penult = []
        for p in G.predecessors(target):
            if p == target:
                continue
            if not G.has_edge(p, target):
                continue
            penult.append((p, GraphTracker._edge_score(G, p, target, greedy_use_abs)))
        penult.sort(key=lambda x: -x[1])

        out = []
        seen_u = set()

        def path_ok(nodes):
            if not nodes or nodes[0] != source or nodes[-1] != target:
                return False
            if len(nodes) - 1 > cutoff:
                return False
            return len(set(nodes)) == len(nodes)

        for U, _ in penult:
            if len(out) >= n_paths:
                break
            if U in seen_u:
                continue

            rev = [target, U]
            cur = U
            visited = set(rev)

            while cur != source and (len(rev) - 1) < cutoff:
                preds = [p for p in G.predecessors(cur) if p not in visited]
                if not preds:
                    break
                best_p = max(preds, key=lambda p: GraphTracker._edge_score(G, p, cur, greedy_use_abs))
                rev.append(best_p)
                visited.add(best_p)
                cur = best_p

            full = list(reversed(rev))
            if not path_ok(full):
                full = None
                try:
                    sp = nx.shortest_path(G, source, U)
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    sp = None
                if sp is not None and sp[-1] == U and target not in sp:
                    cand = sp + [target]
                    if path_ok(cand):
                        full = cand

            if full is not None:
                out.append(full)
                seen_u.add(U)

        return out

    @staticmethod
    def node_first_layer_from_gradients(gradients, modulator_list):
        """Earliest key in gradients.keys() where each gene appears as source or target."""
        layer_keys = sorted(gradients.keys())
        first_layer = {}
        for layer in layer_keys:
            gradient_layer = gradients[layer]
            mod_effect = GraphTracker.get_mod_effect(gradient_layer, modulator_list)
            edge_layer = pd.DataFrame(mod_effect)
            edge_layer.index = edge_layer.index.astype(str).str.replace("beta_", "", regex=False)
            edges = edge_layer.stack().reset_index()
            edges.columns = ["source", "target", "weight"]
            for source, target in zip(edges["source"], edges["target"]):
                if source not in first_layer:
                    first_layer[source] = layer
                if target not in first_layer:
                    first_layer[target] = layer
        return first_layer, layer_keys

    _COLOR_PRESETS = {
        "default": {
            "ligand": "#7fc97f",
            "receptor": "#beaed4",
            "other": "#fdc086",
            "pos": "#386cb0",
            "neg": "#ef3b2c",
        },
        # FOXO1 / full-subgraph style: pink activation, blue inhibition, cool intermediates
        "pathway": {
            "ligand": "#41ab5d",
            "receptor": "#1f5aa6",
            "other": "#9ecae1",
            "pos": "#e377c2",
            "neg": "#4292c6",
        },
    }

    @staticmethod
    def _spring_layout_pinned(subG, source, target, seed=42):
        """Force-directed layout with source / target anchored left and right."""
        n = max(len(subG), 1)
        k = 2.2 / np.sqrt(n)
        rng = np.random.default_rng(seed)
        nodes = list(subG.nodes())
        pos0 = {v: rng.uniform(-0.25, 0.25, size=2) for v in nodes}
        fixed = None
        if source in subG and target in subG and source != target:
            pos0[source] = np.array([-1.35, 0.0])
            pos0[target] = np.array([1.35, 0.0])
            fixed = [source, target]
        return nx.spring_layout(
            subG,
            pos=pos0,
            fixed=fixed,
            k=k,
            iterations=120,
            seed=seed,
            threshold=1e-4,
        )

    @staticmethod
    def _layout_path_columns(path_edges, path_nodes, source, x_step=2.0, y_step=1.05):
        """
        Distinct columns: x = shortest hop count from ``source`` along edges that
        appear on the selected paths (subgraph of path edges only).
        """
        H = nx.DiGraph()
        H.add_nodes_from(path_nodes)
        for (u, v) in path_edges:
            H.add_edge(u, v)
        if not H.nodes():
            return {}
        if source not in H:
            nodes = sorted(H.nodes(), key=str)
            return {n: (0.0, i * y_step) for i, n in enumerate(nodes)}
        lengths = dict(nx.single_source_shortest_path_length(H, source))
        maxd = max(lengths.values()) if lengths else 0
        by_col = {}
        for n in H.nodes():
            col = lengths.get(n, maxd + 1)
            by_col.setdefault(col, []).append(n)
        pos = {}
        for col in sorted(by_col):
            nodes = sorted(by_col[col], key=str)
            n_here = len(nodes)
            for i, n in enumerate(nodes):
                pos[n] = (col * x_step, (i - (n_here - 1) / 2.0) * y_step)
        return pos

    @staticmethod
    def _node_order_vertical(paths, subG, first_layer, default_subset):
        """Single column: walk paths in order, then tie-break by layer and name."""
        ordered = []
        seen = set()
        for path in paths:
            for n in path:
                if n not in seen and n in subG:
                    seen.add(n)
                    ordered.append(n)
        for n in sorted(subG.nodes(), key=str):
            if n not in seen:
                ordered.append(n)
        ly = lambda n: first_layer.get(n, default_subset)
        lens = {}
        if ordered:
            src = ordered[0]
            if src in subG:
                try:
                    und = subG.to_undirected()
                    lens = dict(nx.single_source_shortest_path_length(und, src))
                except nx.NetworkXNoPath:
                    lens = {}
        ordered.sort(key=lambda n: (ly(n), lens.get(n, 9999), str(n)))
        return ordered

    @staticmethod
    def draw_paths_subgraph(
        G,
        paths,
        gradients,
        ligands,
        receptors,
        modulator_list,
        node_size=900,
        font_size=14,
        save_path=None,
        figure_params=None,
        layout="path_columns",
        color_preset="pathway",
        color_dict=None,
        title=None,
        path_source=None,
        path_target=None,
        figsize=(12, 8),
        show_layer_legend=False,
    ):
        """
        Draw highlighted paths on a subgraph.

        Default ``layout="path_columns"``: one column per hop from ``path_source``
        along path edges (Apc at left, Tap1 further right when on longer paths).

        Parameters
        ----------
        layout : str
            ``"path_columns"`` (default), ``"spring"``, ``"vertical"``, ``"multipartite"``,
            or ``"columns"`` (alias for path_columns).
        color_preset : str
            ``"pathway"`` (default) or ``"default"``. Ignored if ``color_dict`` is set.
        path_source, path_target : str, optional
            Path endpoints; ``path_source`` anchors column 0. Passed from ``draw_paths``.
        title : str, optional
            Overrides the default math-style ``source → target`` title.
        show_layer_legend : bool
            If True, add gradient-layer colors to the legend.
        """
        if ligands is None:
            ligands = set()
        if receptors is None:
            receptors = set()
        path_nodes = set()
        path_edges = {}
        for path in paths:
            path_nodes.update(path)
            for u, v in zip(path[:-1], path[1:]):
                path_edges[(u, v)] = G.get_edge_data(u, v, default={})

        if color_dict is None:
            color_dict = dict(
                GraphTracker._COLOR_PRESETS.get(color_preset, GraphTracker._COLOR_PRESETS["default"])
            )

        first_layer, layer_keys = GraphTracker.node_first_layer_from_gradients(
            gradients, modulator_list
        )

        subG = G.subgraph(path_nodes).copy()
        default_subset = (max(layer_keys) + 1) if layer_keys else 0
        for n in subG.nodes():
            subG.nodes[n]["subset"] = first_layer.get(n, default_subset)

        layout_lc = (layout or "path_columns").lower()
        if layout_lc in ("columns", "path_columns", "path"):
            layout_lc = "path_columns"
        ps_col = path_source
        if ps_col is None and paths and paths[0]:
            ps_col = paths[0][0]

        if layout_lc == "multipartite":
            pos = nx.multipartite_layout(subG, subset_key="subset", scale=1.2)
        elif layout_lc == "spring":
            pos = GraphTracker._spring_layout_pinned(
                subG, path_source, path_target, seed=42
            )
        elif layout_lc == "path_columns":
            pos = GraphTracker._layout_path_columns(
                path_edges, path_nodes, ps_col
            )
        else:
            nodes_ord = GraphTracker._node_order_vertical(
                paths, subG, first_layer, default_subset
            )
            nN = len(nodes_ord)
            dy = 1.15
            pos = {
                nodes_ord[i]: (0.0, (nN - 1 - i) * dy) for i in range(nN)
            }

        _, ax = plt.subplots(figsize=figsize)

        node_colors = []
        for node in subG.nodes():
            if node in ligands:
                node_colors.append(color_dict["ligand"])
            elif node in receptors:
                node_colors.append(color_dict["receptor"])
            else:
                node_colors.append(color_dict["other"])

        nx.draw_networkx_nodes(
            subG,
            pos,
            ax=ax,
            node_size=node_size,
            node_color=node_colors,
            edgecolors="black",
            linewidths=0.5,
            alpha=0.88,
        )

        edges_data = [(u, v, path_edges[(u, v)]) for u, v in path_edges]
        edges_weights = []
        edges_colors = []
        for _, _, d in edges_data:
            w = d["weight"] if isinstance(d, dict) and "weight" in d else 1.0
            edges_weights.append(w)
            edges_colors.append(color_dict["neg"] if float(w) < 0 else color_dict["pos"])

        max_w = max(abs(float(w)) for w in edges_weights) if edges_weights else 1.0
        if layout_lc == "spring":
            w_scale = 5.5
        elif layout_lc == "path_columns":
            w_scale = 3.2
        else:
            w_scale = 2.8
        widths = [w_scale * abs(float(w)) / max_w for w in edges_weights]

        for idx, ((u, v, d), color, width) in enumerate(
            zip(edges_data, edges_colors, widths)
        ):
            if layout_lc == "spring":
                rad = 0.28 if idx % 2 == 0 else -0.28
            else:
                rad = 0.32 if idx % 2 == 0 else -0.32
            connstyle = f"arc3,rad={rad}"
            min_w = 0.75 if layout_lc == "spring" else 0.6
            nx.draw_networkx_edges(
                subG,
                pos,
                ax=ax,
                edgelist=[(u, v)],
                width=max(float(width), min_w),
                edge_color=color,
                arrowstyle="-|>",
                arrowsize=18,
                alpha=0.85,
                connectionstyle=connstyle,
                min_source_margin=12,
                min_target_margin=14,
            )

        labels = {node: f"$\\mathit{{{node.capitalize()}}}$" for node in subG.nodes()}
        nx.draw_networkx_labels(subG, pos, labels=labels, font_size=font_size, ax=ax)

        node_patches = []
        if color_dict["ligand"] == color_dict["receptor"]:
            node_patches.append(
                mpatches.Patch(
                    facecolor=color_dict["ligand"],
                    edgecolor="black",
                    label="Ligand / receptor",
                )
            )
        else:
            node_patches.extend(
                [
                    mpatches.Patch(
                        facecolor=color_dict["ligand"],
                        edgecolor="black",
                        label="Ligand",
                    ),
                    mpatches.Patch(
                        facecolor=color_dict["receptor"],
                        edgecolor="black",
                        label="Receptor",
                    ),
                ]
            )
        node_patches.append(
            mpatches.Patch(
                facecolor=color_dict["other"],
                edgecolor="black",
                label="Other gene",
            )
        )

        edge_patches = [
            mpatches.Patch(
                facecolor=color_dict["pos"],
                edgecolor="black",
                label="Positive edge",
            ),
            mpatches.Patch(
                facecolor=color_dict["neg"],
                edgecolor="black",
                label="Negative edge",
            ),
        ]

        extra = []
        if show_layer_legend:
            if len(layer_keys) <= 1:
                layer_frac = [0.5]
            else:
                layer_frac = [i / (len(layer_keys) - 1) for i in range(len(layer_keys))]
            extra = [
                mpatches.Patch(
                    facecolor=plt.cm.viridis(layer_frac[i]),
                    edgecolor="black",
                    label=f"Earliest layer {layer} (gradients key)",
                )
                for i, layer in enumerate(layer_keys)
            ]

        ax.legend(
            handles=node_patches + edge_patches + extra,
            loc="center left",
            bbox_to_anchor=(1.02, 0.5),
            frameon=True,
            fontsize=9,
            ncol=1,
            title="Legend",
            title_fontsize=10,
        )

        layer_str = ", ".join(str(k) for k in layer_keys)
        if title is None:
            ps, pt = path_source, path_target
            if ps is None and paths and paths[0]:
                ps = paths[0][0]
            if pt is None and paths and paths[0]:
                pt = paths[0][-1]
            if layout_lc != "multipartite" and ps is not None and pt is not None:
                title = (
                    f"$\\mathit{{{str(ps).capitalize()}}}$ "
                    f"$\\rightarrow$ $\\mathit{{{str(pt).capitalize()}}}$"
                )
            else:
                title = f"Top paths — layers [{layer_str}]"
        ax.set_title(title, fontsize=14)
        ax.axis("off")
        ax.margins(0.15)
        plt.tight_layout()
        if save_path is not None:
            if figure_params is None:
                figure_params = dict(bbox_inches="tight")
            d = os.path.dirname(save_path)
            if d:
                os.makedirs(d, exist_ok=True)
            plt.savefig(save_path, **figure_params)
        plt.show()

    def build_graph_from_gradients(self, gradients, modulator_list, min_edge_weight=1e-4):
        """
        Build or rebuild self.G from gradients and given modulator_list.
        modulator_list: list of genes to use as modulator columns, e.g. ['Apc', 'Insig1']
        """
        self.G.clear()
        self.gradients = gradients
        self.modulator_list = list(modulator_list)
        for layer, gradient_layer in gradients.items():
            mod_effect = self.get_mod_effect(gradient_layer, self.modulator_list)
            edge_layer = pd.DataFrame(mod_effect)
            edge_layer.index = edge_layer.index.astype(str).str.replace("beta_", "", regex=False)
            edges = edge_layer.stack().reset_index()
            edges.columns = ['source', 'target', 'weight']
            self.G.add_weighted_edges_from(edges.values)

        # Remove low-effect edges:
        edges_to_remove = [
            (u, v) for u, v, d in self.G.edges(data=True)
            if abs(d.get("weight", 0)) < min_edge_weight
        ]
        self.G.remove_edges_from(edges_to_remove)

    def top_paths(
        self,
        source,
        target,
        cutoff,
        n_paths=10,
        path_selection: str = "max_last_edge",
        greedy_use_abs: bool = True,
    ):
        """
        ``path_selection``:

        - ``\"max_last_edge\"`` — :meth:`get_top_paths` (all simple paths, ranked by
          |last-edge weight|).
        - ``\"iterative_distinct_penultimate\"`` — :meth:`get_iterative_distinct_paths`
          (one path per distinct penultimate node, greedy backward from each).
        """
        if path_selection == "max_last_edge":
            return self.get_top_paths(self.G, source, target, cutoff, n_paths)
        if path_selection == "iterative_distinct_penultimate":
            return self.get_iterative_distinct_paths(
                self.G, source, target, cutoff, n_paths, greedy_use_abs=greedy_use_abs
            )
        raise ValueError(
            f"path_selection must be 'max_last_edge' or 'iterative_distinct_penultimate', got {path_selection!r}"
        )

    def draw_paths(
        self,
        source,
        target,
        cutoff,
        n_paths=10,
        ligands=None,
        receptors=None,
        save_path=None,
        gradients=None,
        modulator_list=None,
        path_selection: str = "max_last_edge",
        greedy_use_abs: bool = True,
        **kwargs,
    ):
        """
        Draw a subgraph containing up to ``n_paths`` paths from ``source`` to ``target``.

        ``path_selection``:

        - ``\"max_last_edge\"`` (default) — rank all simple paths by |edge from
          penultimate node → ``target``|; see :meth:`get_top_paths`.
        - ``\"iterative_distinct_penultimate\"`` — for each of the strongest distinct
          edges ``U → target``, trace backward by strongest incoming edges toward
          ``source``; see :meth:`get_iterative_distinct_paths`. Use this to get
          ``n_paths`` different ``U`` tied to ``target``.

        ``greedy_use_abs`` applies only to iterative mode (largest |weight| at each step).

        If ``gradients`` / ``modulator_list`` are supplied, the graph is rebuilt; otherwise
        the graph from the last :meth:`build_graph_from_gradients` call is used.
        """
        # Allow building/rebuilding if gradients and modulator_list supplied
        if gradients is not None and modulator_list is not None:
            self.build_graph_from_gradients(gradients, modulator_list)
        elif self.G is None or len(self.G.nodes) == 0:
            raise RuntimeError("Call build_graph_from_gradients or supply gradients+modulator_list to draw_paths.")

        top_paths = self.top_paths(
            source,
            target,
            cutoff,
            n_paths,
            path_selection=path_selection,
            greedy_use_abs=greedy_use_abs,
        )
        print(f"Top {len(top_paths)} paths ({path_selection}): {top_paths}")

        # Collect modulator_list for the found paths for highlighting and legend coloring
        if modulator_list is None:
            modulator_list_used = []
            for path in top_paths:
                for node in path:
                    if node not in modulator_list_used:
                        modulator_list_used.append(node)
        else:
            modulator_list_used = modulator_list

        # Use stored gradients if not specified at draw-time
        gradients_to_use = gradients if gradients is not None else self.gradients

        # Actually draw (defaults: spring + pathway colors; pass layout= / color_preset= to change)
        kw = dict(path_source=source, path_target=target)
        kw.update(kwargs)
        self.draw_paths_subgraph(
            self.G,
            top_paths,
            gradients_to_use,
            ligands=ligands,
            receptors=receptors,
            modulator_list=modulator_list_used,
            save_path=save_path,
            **kw,
        )

