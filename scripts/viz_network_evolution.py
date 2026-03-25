"""Manim animation: neural network node errors evolving during training.

Shows a network diagram where each activation node is colored by its
gradient magnitude at each training step. One trajectory, animated.

Usage:
    # Synthetic data (small network)
    uv run manim -pql scripts/viz_network_evolution.py NetworkEvolution

    # Real data from gradient_trajectories.npz
    uv run manim -pql scripts/viz_network_evolution.py NetworkEvolution \
        --renderer=opengl -c "DATA_PATH='data/gradient_trajectories.npz'"

    # High quality
    uv run manim -pqh scripts/viz_network_evolution.py NetworkEvolution
"""

import numpy as np
from manim import (
    Scene, VGroup, Circle, Line, Text, Rectangle,
    interpolate_color,
    UP, DOWN, RIGHT, LEFT,
    WHITE, BLUE, RED, GREY,
)

WEIGHT_LABELS_5 = ["u", "u_x", "u_y", "u_xx", "u_yy"]


# ── Data loading ────────────────────────────────────────────────────────

def load_trajectory_from_npz(path: str, traj_idx: int = 0) -> np.ndarray:
    """Load one trajectory's gradient magnitudes from NPZ.

    Args:
        path: Path to gradient_trajectories.npz
        traj_idx: Which trajectory to use

    Returns:
        (n_steps, n_params) array of absolute gradient values.
    """
    data = np.load(path, allow_pickle=True)
    gradients = data["gradients"]  # (n_trajectories, n_steps, n_params)
    return np.abs(gradients[traj_idx])  # (n_steps, n_params)


def generate_synthetic_trajectory(
    layers: list[int],
    n_steps: int = 200,
    seed: int = 42,
) -> tuple[np.ndarray, list[int]]:
    """Generate synthetic gradient magnitudes for one trajectory.

    Returns:
        gradients: (n_steps, n_nodes) array of gradient magnitudes.
        layers: the architecture used.
    """
    rng = np.random.default_rng(seed)
    t = np.arange(n_steps)

    node_grads = []
    for layer_idx in range(1, len(layers)):
        n_nodes = layers[layer_idx]
        base_rate = 0.015 / layer_idx
        base_scale = 0.5 * layer_idx

        for _ in range(n_nodes):
            rate = base_rate * (1 + 0.3 * rng.standard_normal())
            rate = max(rate, 0.002)
            scale = base_scale * (1 + 0.2 * rng.standard_normal())
            gradient = scale * np.exp(-rate * t)
            gradient += rng.standard_normal(n_steps) * 0.02 * scale
            node_grads.append(np.abs(gradient))

    return np.array(node_grads).T, layers  # (n_steps, n_nodes)


# ── Manim scene ─────────────────────────────────────────────────────────

def _build_color_bar(vmin: float, vmax: float, height: float = 3.0,
                     width: float = 0.25, n_segments: int = 20) -> VGroup:
    """Build a vertical color bar: blue (low |∂L/∂w|) → red (high |∂L/∂w|)."""
    bar = VGroup()
    seg_h = height / n_segments

    for i in range(n_segments):
        t = i / (n_segments - 1)
        color = interpolate_color(BLUE, RED, t)
        rect = Rectangle(
            width=width, height=seg_h,
            stroke_width=0, fill_color=color, fill_opacity=1.0,
        )
        y = -(height / 2) + (seg_h / 2) + (i * seg_h)
        rect.move_to([0, y, 0])
        bar.add(rect)

    # Labels
    low_label = Text(f"{vmin:.1e}", font_size=14, color=WHITE)
    low_label.next_to(bar, DOWN, buff=0.1)
    high_label = Text(f"{vmax:.1e}", font_size=14, color=WHITE)
    high_label.next_to(bar, UP, buff=0.1)
    title_label = Text("|∂L/∂w|", font_size=16, color=WHITE)
    title_label.next_to(high_label, UP, buff=0.15)

    bar.add(low_label, high_label, title_label)
    return bar


class NetworkEvolution(Scene):
    """Animate neural network gradient evolution during training."""

    def construct(self):
        import os
        # Load real data via env var, or fall back to synthetic
        data_path = os.environ.get("GRADIENT_DATA", None)

        if data_path:
            grads = load_trajectory_from_npz(data_path)
            n_steps, n_params = grads.shape
            # For cheat model: 5 params = 5 input weights → 1 output
            if n_params == 5:
                layers = [5, 1]
                labels = WEIGHT_LABELS_5
            elif n_params == 10:
                layers = [10, 1]
                labels = [f"w{i}" for i in range(n_params)]
            else:
                layers = [n_params, 1]
                labels = [f"w{i}" for i in range(n_params)]
            # grads are per-weight, show as edge colors instead of node colors
            self._animate_linear(grads, layers, labels, n_steps)
        else:
            # Synthetic: bigger network for visual interest
            layers = [5, 6, 6, 1]
            grads, _ = generate_synthetic_trajectory(layers, n_steps=200)
            n_steps = grads.shape[0]
            self._animate_mlp(grads, layers, n_steps)

    def _animate_linear(self, grads, layers, labels, n_steps):
        """Animate a linear model (no hidden layers)."""
        n_input = layers[0]
        n_output = layers[-1]

        vmin, vmax = float(grads.min()), float(grads.max())

        # Layout: input nodes on left, output on right, labeled edges
        input_circles = []
        x_in, x_out = -3, 3
        spacing = 0.9
        y_start = ((n_input - 1) * spacing) / 2

        all_elements = VGroup()

        for i in range(n_input):
            y = y_start - (i * spacing)
            c = Circle(radius=0.3, stroke_color=WHITE, stroke_width=1.5,
                        fill_color=GREY, fill_opacity=0.4)
            c.move_to([x_in, y, 0])
            label = Text(labels[i], font_size=16, color=WHITE)
            label.next_to(c, direction=[-1, 0, 0], buff=0.15)
            input_circles.append(c)
            all_elements.add(c, label)

        output_circle = Circle(radius=0.4, stroke_color=WHITE, stroke_width=2,
                                fill_color=BLUE, fill_opacity=0.9)
        output_circle.move_to([x_out, 0, 0])
        out_label = Text("u_t", font_size=20, color=WHITE)
        out_label.next_to(output_circle, direction=[1, 0, 0], buff=0.15)
        all_elements.add(output_circle, out_label)

        # Edges (colored by gradient)
        edges = []
        for i in range(n_input):
            line = Line(
                input_circles[i].get_center(), output_circle.get_center(),
                stroke_width=3, stroke_opacity=0.8,
            )
            edges.append(line)
            all_elements.add(line)

        title = Text("Linear Model — Gradient Evolution", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.3)

        step_text = Text("Step: 0", font_size=24, color=WHITE)
        step_text.to_corner(UP + RIGHT, buff=0.5)

        # Color bar
        color_bar = _build_color_bar(vmin, vmax)
        color_bar.to_edge(RIGHT, buff=0.5)

        self.add(all_elements, title, step_text, color_bar)
        self.wait(0.5)

        # Animate: sample keyframes
        keyframe_interval = max(1, n_steps // 150)
        keyframes = list(range(0, n_steps, keyframe_interval))
        if keyframes[-1] != n_steps - 1:
            keyframes.append(n_steps - 1)

        for step in keyframes:
            for i, edge in enumerate(edges):
                val = grads[step, i]
                t_norm = float(np.clip((val - vmin) / (vmax - vmin + 1e-8), 0, 1))
                color = interpolate_color(BLUE, RED, t_norm)
                edge.set_stroke(color=color, width=2 + (6 * t_norm))

            new_text = Text(f"Step: {step}", font_size=24, color=WHITE)
            new_text.to_corner(UP + RIGHT, buff=0.5)
            self.remove(step_text)
            step_text = new_text
            self.add(step_text)
            self.wait(1.0 / 30)

        self.wait(1)

    def _animate_mlp(self, grads, layers, n_steps):
        """Animate an MLP (hidden layers)."""
        vmin, vmax = float(grads.min()), float(grads.max())

        node_circles: dict[tuple[int, int], Circle] = {}
        all_elements = VGroup()

        layer_spacing = 2.5
        total_width = (len(layers) - 1) * layer_spacing
        x_start = -total_width / 2

        # Map from (layer, node) to column in grads
        grad_col = 0
        node_to_col: dict[tuple[int, int], int] = {}
        for layer_idx in range(1, len(layers)):
            for node_idx in range(layers[layer_idx]):
                node_to_col[(layer_idx, node_idx)] = grad_col
                grad_col += 1

        # Create nodes
        for layer_idx, n_nodes in enumerate(layers):
            node_spacing = 0.8
            total_height = (n_nodes - 1) * node_spacing
            y_start = total_height / 2
            x = x_start + (layer_idx * layer_spacing)

            for node_idx in range(n_nodes):
                y = y_start - (node_idx * node_spacing)
                radius = 0.25 if layer_idx > 0 else 0.2
                circle = Circle(radius=radius, stroke_color=WHITE,
                                stroke_width=1.5, fill_opacity=0.9)
                circle.move_to([x, y, 0])
                if layer_idx == 0:
                    circle.set_fill(GREY, opacity=0.4)
                else:
                    circle.set_fill(BLUE, opacity=0.9)
                node_circles[(layer_idx, node_idx)] = circle
                all_elements.add(circle)

        # Connections
        connections = VGroup()
        for layer_idx in range(len(layers) - 1):
            for src in range(layers[layer_idx]):
                for dst in range(layers[layer_idx + 1]):
                    line = Line(
                        node_circles[(layer_idx, src)].get_center(),
                        node_circles[(layer_idx + 1, dst)].get_center(),
                        stroke_color=GREY, stroke_width=0.5, stroke_opacity=0.3,
                    )
                    connections.add(line)

        # Labels
        layer_labels = VGroup()
        for layer_idx, n_nodes in enumerate(layers):
            x = x_start + (layer_idx * layer_spacing)
            if layer_idx == 0:
                lt = "Input"
            elif layer_idx == len(layers) - 1:
                lt = "Output"
            else:
                lt = f"Hidden {layer_idx}"
            label = Text(lt, font_size=18, color=WHITE)
            total_height = (n_nodes - 1) * 0.8
            label.move_to([x, -(total_height / 2) - 0.6, 0])
            layer_labels.add(label)

        title = Text("Gradient Evolution During Training", font_size=28, color=WHITE)
        title.to_edge(UP, buff=0.3)
        step_text = Text("Step: 0", font_size=24, color=WHITE)
        step_text.to_corner(UP + RIGHT, buff=0.5)

        color_bar = _build_color_bar(vmin, vmax)
        color_bar.to_edge(RIGHT, buff=0.5)

        self.add(connections, all_elements, layer_labels, title, step_text, color_bar)
        self.wait(0.5)

        # Animate keyframes
        keyframe_interval = max(1, n_steps // 150)
        keyframes = list(range(0, n_steps, keyframe_interval))
        if keyframes[-1] != n_steps - 1:
            keyframes.append(n_steps - 1)

        for step in keyframes:
            for (layer_idx, node_idx), col in node_to_col.items():
                circle = node_circles[(layer_idx, node_idx)]
                val = grads[step, col]
                t_norm = float(np.clip((val - vmin) / (vmax - vmin + 1e-8), 0, 1))
                color = interpolate_color(BLUE, RED, t_norm)
                circle.set_fill(color, opacity=0.9)

            new_text = Text(f"Step: {step}", font_size=24, color=WHITE)
            new_text.to_corner(UP + RIGHT, buff=0.5)
            self.remove(step_text)
            step_text = new_text
            self.add(step_text)
            self.wait(1.0 / 30)

        self.wait(1)
