#!/usr/bin/env python3
"""Plot training-loss diagnostics across experiments.

Three figures are produced in figures/:
  - training_loss_main_aux.png  mse_main + aux per exp/mixer (log y)
  - training_loss_kendall.png   train_loss totals + s_mse/s_aux log-vars
  - training_loss_ratios.png    per-pde ratio grid respecting
                                precompose < library < raw precedence

Usage:
    python scripts/plot_training_loss.py <exp_dir> [exp_dir ...]
    python scripts/plot_training_loss.py --config configs/.../foo.yaml
"""

import argparse
import colorsys
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.colors as mcolors
import matplotlib.pyplot as plt
import yaml

PRECEDENCE = ["precompose", "library", "raw"]
SMOOTH_WIN = 50
PANEL_W = 5.0
PANEL_H = 3.5
AUX_LINESTYLES = [
    "--",
    ":",
    "-.",
    (0, (5, 1)),
    (0, (3, 1, 1, 1)),
    (0, (1, 1, 5, 1)),
]


@dataclass
class MixerHist:
    mse_main: np.ndarray
    aux: Dict[str, np.ndarray]
    s_mse: np.ndarray
    s_aux: Dict[str, np.ndarray]


@dataclass
class ExpData:
    name: str
    pde_type: str
    input_mode: str
    variant: str
    capacity: int
    iters: np.ndarray
    train_loss: np.ndarray
    mixers: Dict[str, MixerHist]
    lr_curve: Optional[np.ndarray]


def smooth(y: np.ndarray, window: int = SMOOTH_WIN) -> Tuple[np.ndarray, int]:
    if len(y) < window or window <= 1:
        return y, len(y)
    kernel = np.ones(window) / window
    s = np.convolve(y, kernel, mode="valid")
    return s, len(s)


def reconstruct_lr(t: dict, n_iters: int) -> Optional[np.ndarray]:
    if not t.get("use_scheduler"):
        return None
    base = float(t.get("outer_lr", 0.0))
    minl = float(t.get("min_lr", 0.0))
    warmup = int(t.get("warmup_iterations", 0))
    max_iter = int(t.get("max_iterations", n_iters))
    post = max_iter - warmup
    if post <= 0 or base <= 0:
        return None
    stype = t.get("scheduler_type", "constant")
    lr = np.full(n_iters, base, dtype=float)
    for i in range(n_iters):
        step = i
        if stype == "cosine":
            lr[i] = minl + 0.5 * (base - minl) * (1 + np.cos(np.pi * step / post))
        elif stype == "exponential":
            gamma = (minl / base) ** (1.0 / post) if base > 0 else 1.0
            lr[i] = base * (gamma ** step)
        elif stype == "polynomial":
            p = float(t.get("poly_power", 1.0))
            if step < post:
                lr[i] = (base - minl) * ((1 - step / post) ** p) + minl
            else:
                lr[i] = minl
    return lr


def load_exp(exp_dir: Path) -> Optional[ExpData]:
    hp = exp_dir / "training" / "history.json"
    cp = exp_dir / "training" / "config.yaml"
    if not hp.exists():
        print(f"  SKIP {exp_dir}: no history.json")
        return None
    if not cp.exists():
        print(f"  SKIP {exp_dir}: no config.yaml")
        return None
    with open(hp) as f:
        h = json.load(f)
    with open(cp) as f:
        cfg = yaml.safe_load(f)

    pde_type = cfg.get("experiment", {}).get("pde_type", "?")
    t = cfg.get("training", {}) or {}
    input_mode = t.get("input_mode", "library")
    hidden_dims = t.get("hidden_dims") or []
    variant = "x".join(str(d) for d in hidden_dims) if hidden_dims else "?"
    capacity = sum(int(d) for d in hidden_dims) if hidden_dims else 0

    train_loss = np.array(h.get("train_loss", []))
    iters = np.array(h.get("iteration", list(range(len(train_loss)))))

    mixers: Dict[str, MixerHist] = {}
    for m_name, m_data in (h.get("mixers") or {}).items():
        aux = {k: np.array(v) for k, v in (m_data.get("aux") or {}).items()}
        s_aux = {k: np.array(v) for k, v in (m_data.get("s_aux") or {}).items()}
        mixers[m_name] = MixerHist(
            mse_main=np.array(m_data.get("mse_main", [])),
            aux=aux,
            s_mse=np.array(m_data.get("s_mse", [])),
            s_aux=s_aux,
        )

    lr = reconstruct_lr(t, len(iters))
    return ExpData(
        name=exp_dir.name,
        pde_type=pde_type,
        input_mode=input_mode,
        variant=variant,
        capacity=capacity,
        iters=iters,
        train_loss=train_loss,
        mixers=mixers,
        lr_curve=lr,
    )


def _plot_smoothed(ax, x, y, label, color, linestyle="-", linewidth=1.2, alpha=1.0):
    if len(y) == 0:
        return
    s, n = smooth(y)
    if n < len(y):
        ax.plot(x[:n], s, label=label, color=color, linestyle=linestyle,
                linewidth=linewidth, alpha=alpha)
        ax.plot(x, y, color=color, alpha=0.10, linewidth=0.4)
    else:
        ax.plot(x, y, label=label, color=color, linestyle=linestyle,
                linewidth=linewidth, alpha=alpha)


def _overlay_lr(ax, exps, color="gray"):
    ax2 = ax.twinx()
    any_lr = False
    for e in exps:
        if e.lr_curve is not None:
            ax2.plot(
                e.iters[: len(e.lr_curve)], e.lr_curve,
                linestyle=":", linewidth=1.0, color=color, alpha=0.5,
            )
            any_lr = True
    if any_lr:
        ax2.set_ylabel("LR", color=color)
        ax2.set_yscale("log")
        ax2.tick_params(axis="y", labelcolor=color)
    else:
        ax2.axis("off")
    return ax2


def _exp_colors(exps: List[ExpData]) -> Dict[str, str]:
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]
    return {e.name: cycle[i % len(cycle)] for i, e in enumerate(exps)}


def _mixer_variant(base_color, mixer_idx: int):
    """Lightness-shifted variant of base_color so mixers within an exp differ."""
    offsets = [0.0, -0.22, 0.22, -0.40, 0.40]
    r, g, b = mcolors.to_rgb(base_color)
    h, l, s = colorsys.rgb_to_hls(r, g, b)
    delta = offsets[mixer_idx % len(offsets)]
    new_l = max(0.15, min(0.85, l + delta))
    return colorsys.hls_to_rgb(h, new_l, s)


def _aux_style(aux_idx: int):
    return AUX_LINESTYLES[aux_idx % len(AUX_LINESTYLES)]


def plot_main_aux(exps: List[ExpData], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    exp_color = _exp_colors(exps)
    for e in exps:
        for mi, (m_name, mh) in enumerate(e.mixers.items()):
            color = _mixer_variant(exp_color[e.name], mi)
            _plot_smoothed(
                ax, e.iters, mh.mse_main,
                label=f"{e.name} m[{m_name}] mse_main",
                color=color, linestyle="-", linewidth=1.6,
            )
            for ai, (aux_name, vals) in enumerate(sorted(mh.aux.items())):
                _plot_smoothed(
                    ax, e.iters, vals,
                    label=f"{e.name} m[{m_name}] aux:{aux_name}",
                    color=color, linestyle=_aux_style(ai), linewidth=1.0, alpha=0.85,
                )
    _overlay_lr(ax, exps)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss (smoothed, window=50)")
    ax.set_yscale("log")
    ax.set_title("Meta-training: mse_main + aux losses")
    ax.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.10, 1.0))
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def plot_kendall(exps: List[ExpData], out: Path) -> None:
    fig, (ax_top, ax_bot) = plt.subplots(2, 1, figsize=(13, 9), sharex=True)
    exp_color = _exp_colors(exps)

    for e in exps:
        _plot_smoothed(
            ax_top, e.iters, e.train_loss,
            label=e.name, color=exp_color[e.name],
            linestyle="-", linewidth=1.6,
        )
    _overlay_lr(ax_top, exps)
    ax_top.set_ylabel("train_loss (Kendall total, smoothed)")
    ax_top.set_yscale("symlog", linthresh=0.1)
    ax_top.set_title("Kendall total loss")
    ax_top.legend(fontsize=7, loc="best")
    ax_top.grid(True, alpha=0.3)

    for e in exps:
        for mi, (m_name, mh) in enumerate(e.mixers.items()):
            color = _mixer_variant(exp_color[e.name], mi)
            _plot_smoothed(
                ax_bot, e.iters, mh.s_mse,
                label=f"{e.name} m[{m_name}] s_mse",
                color=color, linestyle="-", linewidth=1.4,
            )
            for ai, (aux_name, vals) in enumerate(sorted(mh.s_aux.items())):
                _plot_smoothed(
                    ax_bot, e.iters, vals,
                    label=f"{e.name} m[{m_name}] s_aux:{aux_name}",
                    color=color, linestyle=_aux_style(ai), linewidth=0.9, alpha=0.85,
                )
    _overlay_lr(ax_bot, exps)
    ax_bot.set_xlabel("Iteration")
    ax_bot.set_ylabel("Kendall log-variances (s_*)")
    ax_bot.set_title("Per-loss log-variance trajectories")
    ax_bot.legend(fontsize=6, loc="upper left", bbox_to_anchor=(1.10, 1.0))
    ax_bot.grid(True, alpha=0.3)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def discover_ratio_pairs(
    exps: List[ExpData],
) -> List[Tuple[str, ExpData, ExpData, str]]:
    """Pair (e1, e2) is valid when same pde_type and exactly one of
    (input_mode, variant) differs. Numerator is the higher-expected-loss side:
      - mode-axis: precompose < library < raw → num is later in PRECEDENCE
      - variant-axis: smaller capacity → higher loss → num has smaller capacity
    Returns (pde, num, den, axis) with axis ∈ {"mode", "variant"}.
    """
    panels: List[Tuple[str, ExpData, ExpData, str]] = []
    by_pde: Dict[str, List[ExpData]] = {}
    for e in exps:
        by_pde.setdefault(e.pde_type, []).append(e)

    for pde, group in by_pde.items():
        for i in range(len(group)):
            for j in range(i + 1, len(group)):
                a, b = group[i], group[j]
                same_mode = a.input_mode == b.input_mode
                same_variant = a.variant == b.variant
                if same_mode == same_variant:
                    continue
                if not same_mode and same_variant:
                    if a.input_mode not in PRECEDENCE or b.input_mode not in PRECEDENCE:
                        continue
                    if PRECEDENCE.index(a.input_mode) > PRECEDENCE.index(b.input_mode):
                        num, den = a, b
                    else:
                        num, den = b, a
                    panels.append((pde, num, den, "mode"))
                else:
                    if a.capacity < b.capacity:
                        num, den = a, b
                    elif b.capacity < a.capacity:
                        num, den = b, a
                    else:
                        continue
                    panels.append((pde, num, den, "variant"))
    return panels


def plot_ratios(exps: List[ExpData], out: Path) -> None:
    panels = discover_ratio_pairs(exps)
    if not panels:
        print("  (no ratio pairs — need >=2 input_modes within some pde_type)")
        return
    ncols = 2 if len(panels) > 1 else 1
    nrows = (len(panels) + ncols - 1) // ncols
    fig, axes = plt.subplots(
        nrows, ncols,
        figsize=(PANEL_W * ncols, PANEL_H * nrows),
        squeeze=False,
    )
    cycle = plt.rcParams["axes.prop_cycle"].by_key()["color"]

    for idx, (pde, e_num, e_den, axis) in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        if axis == "mode":
            num_lab = f"{e_num.input_mode} [{e_num.variant}]"
            den_lab = f"{e_den.input_mode} [{e_den.variant}]"
        else:
            num_lab = f"{e_num.variant} [{e_num.input_mode}]"
            den_lab = f"{e_den.variant} [{e_den.input_mode}]"
        title = f"{pde}: {num_lab} / {den_lab}"
        ylabel = f"{num_lab} / {den_lab}"

        common_mixers = sorted(set(e_num.mixers.keys()) & set(e_den.mixers.keys()))
        if not common_mixers:
            ax.set_title(f"{title}\n(no shared mixers)")
            ax.axis("off")
            continue
        for mi, m_name in enumerate(common_mixers):
            num_y = e_num.mixers[m_name].mse_main
            den_y = e_den.mixers[m_name].mse_main
            n = min(len(num_y), len(den_y))
            if n == 0:
                continue
            num_s, n_num = smooth(num_y[:n])
            den_s, n_den = smooth(den_y[:n])
            n_s = min(n_num, n_den)
            x = e_num.iters[:n_s]
            ratio = num_s[:n_s] / np.maximum(den_s[:n_s], 1e-30)
            ax.plot(
                x, ratio,
                label=f"mixer {m_name}",
                color=cycle[mi % len(cycle)], linewidth=1.6,
            )
        ax.set_yscale("log")
        ax.set_xlabel("Iteration")
        ax.set_ylabel(ylabel)
        ax.set_title(title)
        ax.axhline(1.0, color="gray", linestyle=":", linewidth=0.8, alpha=0.5)
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    for idx in range(len(panels), nrows * ncols):
        r, c = divmod(idx, ncols)
        axes[r][c].axis("off")

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    print(f"  wrote {out}")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        "exp_dirs", nargs="*",
        help="Experiment directories (containing training/history.json)",
    )
    parser.add_argument(
        "--config", type=str,
        help="Config YAML — resolves to its own experiment dir",
    )
    args = parser.parse_args()

    exp_dirs: List[Path] = []
    if args.config:
        from src.config import ExperimentConfig

        cfg = ExperimentConfig.from_yaml(args.config)
        exp_dirs.append(cfg.exp_dir)
    for d in args.exp_dirs:
        exp_dirs.append(Path(d))

    if not exp_dirs:
        parser.print_help()
        sys.exit(1)

    exps: List[ExpData] = []
    for d in exp_dirs:
        e = load_exp(d)
        if e is not None:
            exps.append(e)

    if not exps:
        print("No usable experiments — nothing to plot.")
        sys.exit(1)

    out_dir = Path("figures")
    out_dir.mkdir(exist_ok=True)
    plot_main_aux(exps, out_dir / "training_loss_main_aux.png")
    plot_kendall(exps, out_dir / "training_loss_kendall.png")
    plot_ratios(exps, out_dir / "training_loss_ratios.png")


if __name__ == "__main__":
    main()
