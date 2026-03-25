"""Interactive 3D gradient surface visualization.

Pipeline:
1. Generate synthetic per-weight gradient histories (or load real data)
2. Aggregate per activation node (mean gradient of incoming weights)
3. SVD per node → extract v_1 fingerprint
4. Hierarchical clustering on v_1s → C groups
5. Aggregate per group → C interactive 3D surfaces (plotly)

Usage:
    uv run python scripts/viz_gradient_surfaces.py
"""

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist


# ── Synthetic data generation ───────────────────────────────────────────

def generate_synthetic_gradients(
    layers: list[int],
    n_trajectories: int = 500,
    n_steps: int = 300,
    seed: int = 42,
) -> dict[tuple[int, int], np.ndarray]:
    """Generate synthetic per-node mean gradient histories.

    Simulates an MLP where:
    - Earlier layers have faster gradient decay (closer to loss)
    - Nodes within a layer have similar but not identical patterns
    - Trajectories (different SGD orderings) diverge over time

    Returns:
        Dict mapping (layer_idx, node_idx) → (n_trajectories, n_steps) array.
        layer_idx starts at 1 (first layer with parameters).
    """
    rng = np.random.default_rng(seed)
    node_gradients: dict[tuple[int, int], np.ndarray] = {}
    t = np.arange(n_steps)[None, :]  # (1, n_steps)

    for layer_idx in range(1, len(layers)):
        n_nodes = layers[layer_idx]

        # Layer-level characteristics
        # Deeper layers: slower decay, larger initial gradients
        base_rate = 0.008 / layer_idx
        base_scale = 0.3 * layer_idx

        for node_idx in range(n_nodes):
            # Per-node variation in decay and scale
            node_rate = base_rate * (1 + 0.4 * rng.standard_normal())
            node_rate = max(node_rate, 0.001)
            node_scale = base_scale * (1 + 0.3 * rng.standard_normal())

            # Base gradient: exponential decay with slight oscillation
            base = node_scale * np.exp(-node_rate * t)

            # Per-trajectory divergence: noise that accumulates
            noise_scale = 0.05 * node_scale
            noise = rng.standard_normal((n_trajectories, n_steps)) * noise_scale
            # Cumulative drift so trajectories actually separate
            drift = np.cumsum(noise * 0.01, axis=1)

            # Per-trajectory initial offset (different starting conditions)
            init_offset = rng.standard_normal((n_trajectories, 1)) * 0.15 * node_scale
            init_decay = np.exp(-0.02 * t)

            gradients = base + drift + (init_offset * init_decay)
            node_gradients[(layer_idx, node_idx)] = gradients

    return node_gradients


# ── SVD fingerprinting ──────────────────────────────────────────────────

def extract_fingerprints(
    node_gradients: dict[tuple[int, int], np.ndarray],
) -> tuple[list[tuple[int, int]], np.ndarray]:
    """SVD each node's surface, extract v_1 as fingerprint.

    Returns:
        keys: list of (layer_idx, node_idx)
        fingerprints: (n_nodes, n_steps) array — one v_1 per node
    """
    keys = sorted(node_gradients.keys())
    fingerprints = []

    for key in keys:
        M = node_gradients[key]  # (n_trajectories, n_steps)
        # Center the data (subtract mean trajectory)
        M_centered = M - M.mean(axis=0, keepdims=True)
        _, _, Vt = np.linalg.svd(M_centered, full_matrices=False)
        v1 = Vt[0]  # first right singular vector (n_steps,)
        fingerprints.append(v1)

    return keys, np.array(fingerprints)


# ── Hierarchical clustering ────────────────────────────────────────────

def cluster_fingerprints(
    fingerprints: np.ndarray,
    keys: list[tuple[int, int]],
    distance_threshold: float = 0.5,
) -> tuple[np.ndarray, int, np.ndarray]:
    """Cluster fingerprints using hierarchical clustering with cosine distance.

    Returns:
        labels: (n_nodes,) cluster assignment
        n_clusters: number of clusters found
        Z: linkage matrix
    """
    # Cosine distance between fingerprints
    distances = pdist(fingerprints, metric="cosine")
    Z = linkage(distances, method="ward")

    # Cut at threshold
    labels = fcluster(Z, t=distance_threshold, criterion="distance")
    n_clusters = len(set(labels))

    return labels, n_clusters, Z


# ── Aggregation and plotting ───────────────────────────────────────────

def aggregate_per_cluster(
    node_gradients: dict[tuple[int, int], np.ndarray],
    keys: list[tuple[int, int]],
    labels: np.ndarray,
    n_clusters: int,
) -> list[np.ndarray]:
    """Mean gradient surface per cluster.

    Returns:
        List of C surfaces, each (n_trajectories, n_steps).
    """
    cluster_surfaces = []
    for c in range(1, n_clusters + 1):
        mask = labels == c
        cluster_keys = [keys[i] for i in range(len(keys)) if mask[i]]
        surfaces = [node_gradients[k] for k in cluster_keys]
        mean_surface = np.mean(surfaces, axis=0)
        cluster_surfaces.append(mean_surface)
    return cluster_surfaces


def sort_mountain(surface: np.ndarray) -> np.ndarray:
    """Sort trajectories at t=0 into a mountain (ascending then descending).

    Args:
        surface: (n_trajectories, n_steps)

    Returns:
        Reindexed surface with mountain shape at t=0.
    """
    t0_values = surface[:, 0]
    sorted_idx = np.argsort(t0_values)

    # Split into ascending (left) and descending (right) halves
    n = len(sorted_idx)
    mid = n // 2
    # Ascending: take every other from sorted (low values)
    # Descending: take remaining in reverse
    left = sorted_idx[::2]       # even indices from sorted
    right = sorted_idx[1::2][::-1]  # odd indices from sorted, reversed
    mountain_idx = np.concatenate([left, right])

    return surface[mountain_idx]


def plot_surface_pair(surface: np.ndarray, title: str):
    """Plot gradient surface and its time derivative side by side."""
    surface = sort_mountain(surface)
    deriv = np.gradient(surface, axis=1)  # ∂/∂t along time axis

    n_traj, n_steps = surface.shape
    traj_idx = np.arange(n_traj)
    step_idx = np.arange(n_steps)
    X, Z = np.meshgrid(traj_idx, step_idx, indexing="ij")

    fig = make_subplots(
        rows=1, cols=2,
        specs=[[{"type": "surface"}, {"type": "surface"}]],
        subplot_titles=["Gradient (∂L/∂w)", "Rate of Change (∂²L/∂w∂t)"],
    )

    fig.add_trace(go.Surface(
        x=X, y=Z, z=surface,
        colorscale="RdBu_r",
        colorbar=dict(title="∂L/∂w", x=0.45),
        showscale=True,
    ), row=1, col=1)

    fig.add_trace(go.Surface(
        x=X, y=Z, z=deriv,
        colorscale="RdBu_r",
        colorbar=dict(title="∂²L/∂w∂t", x=1.0),
        showscale=True,
    ), row=1, col=2)

    scene_common = dict(
        xaxis_title="Trajectory Index",
        yaxis_title="Training Step",
    )
    fig.update_layout(
        title_text=title,
        width=1600, height=700,
        scene={**scene_common, "zaxis_title": "Gradient"},
        scene2={**scene_common, "zaxis_title": "Rate of Change"},
    )
    fig.show()


def plot_surfaces(cluster_surfaces: list[np.ndarray], labels: np.ndarray,
                  keys: list[tuple[int, int]]):
    """Interactive plotly 3D surface pairs — one per cluster."""
    n_clusters = len(cluster_surfaces)

    for c_idx, surface in enumerate(cluster_surfaces):
        cluster_mask = labels == (c_idx + 1)
        cluster_keys = [keys[i] for i in range(len(keys)) if cluster_mask[i]]
        layer_counts: dict[int, int] = {}
        for layer, node in cluster_keys:
            layer_counts[layer] = layer_counts.get(layer, 0) + 1
        layer_desc = ", ".join(f"L{l}: {n} nodes" for l, n in sorted(layer_counts.items()))
        plot_surface_pair(surface, f"Cluster {c_idx + 1}/{n_clusters} — {layer_desc}")


def plot_dendrogram(Z: np.ndarray, keys: list[tuple[int, int]],
                    labels_array: np.ndarray):
    """Interactive plotly dendrogram with color-coded layers."""
    from scipy.cluster.hierarchy import dendrogram as dendro

    # Compute dendrogram data (don't plot)
    ddata = dendro(Z, no_plot=True)

    # Color by layer for leaf labels
    layer_colors = {1: "#636EFA", 2: "#EF553B", 3: "#00CC96"}
    leaf_labels = [f"L{keys[i][0]}:n{keys[i][1]}" for i in ddata["leaves"]]
    leaf_colors = [layer_colors.get(keys[i][0], "#999") for i in ddata["leaves"]]

    fig = go.Figure()

    # Draw dendrogram links
    for i in range(len(ddata["icoord"])):
        x = ddata["icoord"][i]
        y = ddata["dcoord"][i]
        fig.add_trace(go.Scatter(
            x=x, y=y, mode="lines",
            line=dict(color="#555", width=1),
            showlegend=False, hoverinfo="skip",
        ))

    # Draw leaf markers
    fig.add_trace(go.Scatter(
        x=list(range(5, (len(leaf_labels)) * 10 + 1, 10)),
        y=[0] * len(leaf_labels),
        mode="markers+text",
        marker=dict(size=5, color=leaf_colors),
        text=leaf_labels,
        textposition="bottom center",
        textfont=dict(size=7),
        showlegend=False,
        hovertext=[f"{l} (cluster {int(labels_array[ddata['leaves'][i]])})"
                   for i, l in enumerate(leaf_labels)],
        hoverinfo="text",
    ))

    # Layer legend
    for layer, color in layer_colors.items():
        fig.add_trace(go.Scatter(
            x=[None], y=[None], mode="markers",
            marker=dict(size=8, color=color),
            name=f"Layer {layer}",
        ))

    fig.update_layout(
        title="Hierarchical Clustering of Gradient Fingerprints (v₁)",
        xaxis_title="Nodes",
        yaxis_title="Distance",
        width=1200, height=600,
        yaxis=dict(type="log"),  # log scale to see bottom structure
        xaxis=dict(showticklabels=False),
    )
    fig.show()


# ── Data loading ────────────────────────────────────────────────────────

WEIGHT_LABELS_5 = ["u", "u_x", "u_y", "u_xx", "u_yy"]
WEIGHT_LABELS_10 = ["u", "v", "u_x", "u_y", "u_xx", "u_yy",
                    "v_x", "v_y", "v_xx", "v_yy"]


def load_gradient_data(path: str) -> tuple[dict[tuple[int, int], np.ndarray], list[str]]:
    """Load gradient trajectories from NPZ and return per-weight surfaces.

    For a linear model (5 params), returns 5 surfaces keyed as (0, i) for weight i.
    """
    data = np.load(path, allow_pickle=True)
    gradients = data["gradients"]  # (n_trajectories, n_steps, n_params)
    n_traj, n_steps, n_params = gradients.shape
    print(f"  Loaded: {n_traj} trajectories × {n_steps} steps × {n_params} params")

    labels = (
        WEIGHT_LABELS_5 if n_params == 5
        else WEIGHT_LABELS_10 if n_params == 10
        else [f"w{i}" for i in range(n_params)]
    )

    node_gradients: dict[tuple[int, int], np.ndarray] = {}
    for i in range(n_params):
        node_gradients[(0, i)] = gradients[:, :, i]  # (n_traj, n_steps)

    return node_gradients, labels


# ── Main ────────────────────────────────────────────────────────────────

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Interactive 3D gradient surfaces")
    parser.add_argument("--data", type=str, default=None,
                        help="Path to gradient_trajectories.npz (omit for synthetic)")
    parser.add_argument("--no-cluster", action="store_true",
                        help="Skip clustering, show one surface per weight")
    parser.add_argument("--synthetic-layers", type=str, default="5,100,100,1",
                        help="Network shape for synthetic data (comma-separated)")
    parser.add_argument("--synthetic-traj", type=int, default=500)
    parser.add_argument("--synthetic-steps", type=int, default=300)
    args = parser.parse_args()

    # Load or generate data
    weight_labels = None
    if args.data:
        print(f"Loading data from {args.data}...")
        node_gradients, weight_labels = load_gradient_data(args.data)
    else:
        layers = [int(x) for x in args.synthetic_layers.split(",")]
        print(f"Generating synthetic data: {' → '.join(map(str, layers))}")
        print(f"  Trajectories: {args.synthetic_traj}, Steps: {args.synthetic_steps}")
        node_gradients = generate_synthetic_gradients(
            layers, args.synthetic_traj, args.synthetic_steps
        )

    print(f"  {len(node_gradients)} nodes/weights")
    print()

    if args.no_cluster:
        # No clustering — one surface per weight
        keys = sorted(node_gradients.keys())
        surfaces = [node_gradients[k] for k in keys]
        # Fake labels: each weight is its own cluster
        labels = np.arange(1, len(keys) + 1)
        n_clusters = len(keys)

        if weight_labels:
            for surface, label in zip(surfaces, weight_labels):
                plot_surface_pair(surface, f"Weight: {label} (∂L/∂w_{label})")
        else:
            plot_surfaces(surfaces, labels, keys)

    else:
        # Full pipeline: SVD → cluster → aggregate
        print("Extracting v₁ fingerprints via SVD...")
        keys, fingerprints = extract_fingerprints(node_gradients)
        print(f"  Fingerprint shape: {fingerprints.shape}")

        print("Clustering fingerprints...")
        labels, n_clusters, Z = cluster_fingerprints(fingerprints, keys)
        print(f"  Found {n_clusters} clusters")

        for c in range(1, n_clusters + 1):
            mask = labels == c
            cluster_keys = [keys[i] for i in range(len(keys)) if mask[i]]
            layer_counts: dict[int, int] = {}
            for layer, node in cluster_keys:
                layer_counts[layer] = layer_counts.get(layer, 0) + 1
            desc = ", ".join(f"L{l}: {n}" for l, n in sorted(layer_counts.items()))
            print(f"    Cluster {c}: {sum(mask)} nodes ({desc})")

        print("Aggregating surfaces per cluster...")
        cluster_surfaces = aggregate_per_cluster(
            node_gradients, keys, labels, n_clusters
        )

        print("Plotting dendrogram...")
        plot_dendrogram(Z, keys, labels)

        print("Plotting interactive 3D surfaces...")
        plot_surfaces(cluster_surfaces, labels, keys)


if __name__ == "__main__":
    main()
