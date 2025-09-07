import math
import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

def roots_to_networkx(left_rsa=None, right_rsa=None, *, connect_laterals=True):
    """
    Convert one or both split-root RSA objects to a NetworkX Graph.

    Parameters
    ----------
    left_rsa, right_rsa : Root or None
        The left/right Root objects returned by splitrsml2root(...).
    connect_laterals : bool (default True)
        If True, connect the first node of each lateral to the nearest primary node
        (within the same side) so the result is a single connected component per side.

    Returns
    -------
    G : networkx.Graph
        Graph whose nodes have attributes:
            x, y, time, hr, side ('left'/'right'), order ('primary'/'lateral')
        Edges connect consecutive nodes along primary and lateral paths.
        If connect_laterals=True, an extra edge connects each lateral base to primary.
    """
    G = nx.Graph()

    def _add_branch_as_path(G, nodes, *, side, order, branch_index=None):
        """
        Adds a branch (primary or one lateral) as a path of nodes to G.
        Returns the list of (node_id, node_obj) pairs added, in order.
        """
        added = []
        prev_id = None

        for i, node in enumerate(nodes):
            # Stable, unique id: (side, order, branch_idx_or_0, node_index)
            # For primary, branch_index is None -> use 0 for id purposes.
            bid = 0 if branch_index is None else int(branch_index)
            node_id = (side, order, bid, i)

            G.add_node(
                node_id,
                x=float(node.x),
                y=float(node.y),
                time=float(node.time) if node.time is not None else None,
                hr=float(node.hr) if node.hr is not None else None,
                side=side,
                order=order,
            )
            if prev_id is not None:
                G.add_edge(prev_id, node_id, side=side, order=order)
            prev_id = node_id
            added.append((node_id, node))

        return added

    def _add_root(G, root, side: str):
        # Primary path
        primary_pairs = _add_branch_as_path(
            G, root.primary.nodes, side=side, order="primary", branch_index=None
        )

        # Laterals
        if root.laterals:
            for k, lr in enumerate(root.laterals):
                lateral_pairs = _add_branch_as_path(
                    G, lr.nodes, side=side, order="lateral", branch_index=k
                )
                if connect_laterals and lateral_pairs:
                    # Connect the first lateral node to the *nearest* primary node.
                    lx, ly = lateral_pairs[0][1].x, lateral_pairs[0][1].y
                    # Find nearest primary node by Euclidean distance
                    pid_min, _n = min(
                        primary_pairs,
                        key=lambda p: (p[1].x - lx) ** 2 + (p[1].y - ly) ** 2
                    )
                    G.add_edge(
                        pid_min,
                        lateral_pairs[0][0],
                        side=side,
                        order="attachment"
                    )

    if left_rsa is not None:
        _add_root(G, left_rsa, side="left")
    if right_rsa is not None:
        _add_root(G, right_rsa, side="right")

    return G


def plot_rsmlt(G, plate, save_name=None, *, color_attr="hr", flip_y=True, cmap=plt.cm.magma):
    """
    Plot a root graph (from roots_to_networkx) with nodes color-coded by a temporal attribute.

    Parameters
    ----------
    G : networkx.Graph
        Output of roots_to_networkx(...). Nodes must have 'x' and 'y' attrs;
        recommended temporal attrs are 'hr' (default) or 'time'.
    plate : str or int
        Plate identifier for the title.
    save_name : str or None
        If provided, saves the figure to this path.
    color_attr : str
        Node attribute to color by ('hr' by default; 'time' also works).
    flip_y : bool
        If True, plot with negative Y to match image coordinates.
    cmap : matplotlib colormap
        Colormap for the node colors (default: magma).
    """

    # Build positions and colors from node attributes
    node_positions = {}
    node_color = []
    xs, ys = [], []

    for n, d in G.nodes(data=True):
        x = float(d.get("x", np.nan))
        y = float(d.get("y", np.nan))
        if flip_y:
            y = -y

        node_positions[n] = (x, y)
        xs.append(x); ys.append(y)

        cval = d.get(color_attr, None)
        # allow None -> NaN so colorbar still renders
        node_color.append(np.nan if cval is None else float(cval))

    # Compute aspect ratio from bounding box (fallbacks to 1 if degenerate)
    x_range = (np.nanmax(xs) - np.nanmin(xs)) if xs else 1.0
    y_range = (np.nanmax(ys) - np.nanmin(ys)) if ys else 1.0
    ratio = x_range / y_range if y_range not in (0.0, np.nan) else 1.0
    if not np.isfinite(ratio) or ratio <= 0:
        ratio = 1.0

    fig, ax = plt.subplots(figsize=(14, 10/ratio))

    # Draw edges first (thin lines), then colorized nodes
    nx.draw(
        G, pos=node_positions, node_size=0, width=0.7, with_labels=False, ax=ax
    )
    nodes_scatter = nx.draw_networkx_nodes(
        G, pos=node_positions, node_size=10, node_color=node_color, cmap=cmap, ax=ax
    )

    # Title & colorbar labeling
    title = f"RSML + t, plate {plate}"
    ax.set_title(title, fontsize=20)

    # Choose a decent colorbar label
    label_map = {
        "hr": "appearance time (hr)",
        "time": "appearance time (imaged units)",
    }
    cbar_label = label_map.get(color_attr, f"{color_attr}")
    cbar = plt.colorbar(nodes_scatter, aspect=50, shrink=0.5, label=cbar_label, orientation="vertical")
    cbar.set_ticks([])  # Hide tick marks & labels

    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    if save_name is not None:
        fig.savefig(save_name, dpi=300, bbox_inches="tight")

def extract_hr0_graph(G, *, hr_attr="hr", atol=1e-9):
    """
    Return an induced subgraph of nodes whose `hr_attr` is (approximately) 0.

    Parameters
    ----------
    G : networkx.Graph
        Graph from roots_to_networkx(...).
    hr_attr : str
        Node attribute that stores corrected hours (default: 'hr').
    atol : float
        Absolute tolerance for comparing hr to 0.0.

    Returns
    -------
    G0 : networkx.Graph
        Induced subgraph with nodes at hr≈0. Also stores 'subset_ratio' in G0.graph.
    """
    nodes_hr0 = [n for n, d in G.nodes(data=True)
                 if d.get(hr_attr) is not None and np.isclose(float(d[hr_attr]), 0.0, atol=atol)]

    G0 = G.subgraph(nodes_hr0).copy()

    # Compute subset_ratio = (x_range / y_range) on the hr==0 subset
    xs = [float(G0.nodes[n].get("x", np.nan)) for n in G0.nodes]
    ys = [float(G0.nodes[n].get("y", np.nan)) for n in G0.nodes]

    if len(xs) == 0 or len(ys) == 0:
        subset_ratio = 1.0
    else:
        x_rng = np.nanmax(xs) - np.nanmin(xs)
        y_rng = np.nanmax(ys) - np.nanmin(ys)
        subset_ratio = (x_rng / y_rng) if (y_rng not in (0.0, np.nan)) else 1.0
        if not np.isfinite(subset_ratio) or subset_ratio <= 0:
            subset_ratio = 1.0

    G0.graph["subset_ratio"] = subset_ratio
    return G0


def plot_hr0(
    G0,
    plate,
    save_name=None,
    *,
    flip_y=True,
    primary_color="purple",
    lateral_color="crimson",
    node_size=12,
    edge_width=0.8
):
    """
    Plot only the hr==0 graph, with primary nodes purple and laterals crimson.
    Figure size is (5, 5/subset_ratio), where subset_ratio is x_range/y_range on day 0.

    Parameters
    ----------
    G0 : networkx.Graph
        Output of extract_hr0_graph(...). Nodes must have 'x','y','order'.
    plate : str|int
        Plate identifier for the title.
    save_name : str|None
        If provided, saves the figure to this path.
    flip_y : bool
        If True, plot with negative Y to match image coordinates.
    primary_color, lateral_color : str
        Colors for primary and lateral nodes.
    node_size : int
        Node marker size.
    edge_width : float
        Edge linewidth.
    """
    if G0.number_of_nodes() == 0:
        raise ValueError("No hr==0 nodes found in the graph.")

    # Positions (don’t use subset_ratio for pos; it’s only for figsize)
    pos = {}
    xs, ys = [], []
    for n, d in G0.nodes(data=True):
        x = float(d.get("x", np.nan))
        y = float(d.get("y", np.nan))
        ys.append(y); xs.append(x)
        if flip_y:
            y = -y
        pos[n] = (x, y)

    # Figure size: (5, 5/subset_ratio)
    subset_ratio = G0.graph.get("subset_ratio", None)
    if subset_ratio is None or not np.isfinite(subset_ratio) or subset_ratio <= 0:
        # Fallback compute if missing
        x_rng = np.nanmax(xs) - np.nanmin(xs)
        y_rng = np.nanmax(ys) - np.nanmin(ys)
        subset_ratio = (x_rng / y_rng) if (y_rng not in (0.0, np.nan)) else 1.0
        if not np.isfinite(subset_ratio) or subset_ratio <= 0:
            subset_ratio = 1.0

    fig, ax = plt.subplots(figsize=(5, 5 / subset_ratio))

    # Draw edges
    nx.draw_networkx_edges(G0, pos=pos, width=edge_width, ax=ax)

    # Separate nodes by order for color
    primary_nodes = [n for n, d in G0.nodes(data=True) if d.get("order") == "primary"]
    lateral_nodes = [n for n, d in G0.nodes(data=True) if d.get("order") == "lateral"]

    if primary_nodes:
        nx.draw_networkx_nodes(
            G0, pos=pos, nodelist=primary_nodes, node_size=node_size, node_color=primary_color, ax=ax
        )
    if lateral_nodes:
        nx.draw_networkx_nodes(
            G0, pos=pos, nodelist=lateral_nodes, node_size=node_size, node_color=lateral_color, ax=ax
        )

    ax.set_title(f"Day 0 (hr=0) roots, plate {plate}", fontsize=14)
    ax.set_aspect("equal", adjustable="box")
    ax.set_axis_off()
    plt.tight_layout()
    plt.show()

    if save_name is not None:
        fig.savefig(save_name, dpi=300, bbox_inches="tight")
