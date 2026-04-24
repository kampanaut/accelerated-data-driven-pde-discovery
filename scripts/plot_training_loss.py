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
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import yaml

PRECEDENCE = ["precompose", "library", "raw"]
SMOOTH_WIN = 50
PANEL_W = 5.0
PANEL_H = 3.5


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


def plot_main_aux(exps: List[ExpData], out: Path) -> None:
    fig, ax = plt.subplots(figsize=(13, 7))
    exp_color = _exp_colors(exps)
    for e in exps:
        for m_name, mh in e.mixers.items():
            color = exp_color[e.name]
            _plot_smoothed(
                ax, e.iters, mh.mse_main,
                label=f"{e.name} m[{m_name}] mse_main",
                color=color, linestyle="-", linewidth=1.6,
            )
            for aux_name, vals in mh.aux.items():
                _plot_smoothed(
                    ax, e.iters, vals,
                    label=f"{e.name} m[{m_name}] aux:{aux_name}",
                    color=color, linestyle="--", linewidth=1.0, alpha=0.7,
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
        for m_name, mh in e.mixers.items():
            color = exp_color[e.name]
            _plot_smoothed(
                ax_bot, e.iters, mh.s_mse,
                label=f"{e.name} m[{m_name}] s_mse",
                color=color, linestyle="-", linewidth=1.4,
            )
            for aux_name, vals in mh.s_aux.items():
                _plot_smoothed(
                    ax_bot, e.iters, vals,
                    label=f"{e.name} m[{m_name}] s_aux:{aux_name}",
                    color=color, linestyle="--", linewidth=0.9, alpha=0.7,
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
) -> List[Tuple[str, ExpData, ExpData]]:
    by_pde: Dict[str, Dict[str, ExpData]] = {}
    for e in exps:
        by_pde.setdefault(e.pde_type, {})[e.input_mode] = e
    panels: List[Tuple[str, ExpData, ExpData]] = []
    for pde, modes in by_pde.items():
        for j in range(len(PRECEDENCE)):
            for k in range(j):
                num_mode = PRECEDENCE[j]
                den_mode = PRECEDENCE[k]
                if num_mode in modes and den_mode in modes:
                    panels.append((pde, modes[num_mode], modes[den_mode]))
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

    for idx, (pde, e_num, e_den) in enumerate(panels):
        r, c = divmod(idx, ncols)
        ax = axes[r][c]
        common_mixers = sorted(set(e_num.mixers.keys()) & set(e_den.mixers.keys()))
        if not common_mixers:
            ax.set_title(f"{pde}: {e_num.input_mode} / {e_den.input_mode}\n(no shared mixers)")
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
        ax.set_ylabel(f"{e_num.input_mode} / {e_den.input_mode}")
        ax.set_title(f"{pde}: {e_num.input_mode} / {e_den.input_mode}")
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
