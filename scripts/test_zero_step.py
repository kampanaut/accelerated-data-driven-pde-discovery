"""
Test: does MAML's θ* predict the cross-task mean u_t at 0 adaptation steps?

Loads a trained MAML checkpoint, evaluates on collocation points from each
test task with NO inner loop adaptation. Scatter plots pred_u_t vs true_u_t
for both trained θ* (MAML) and initial θ₀ (baseline).

Usage:
    uv run scripts/test_zero_step.py <experiment_dir>
"""

import sys
from pathlib import Path

import numpy as np
import torch
import matplotlib.pyplot as plt
import yaml

from src.networks.pde_operator_network import NetworkConfig, PDEOperatorNetwork
from src.training.task_loader import TASK_REGISTRY


def load_models(exp_dir: Path, device: str = "cpu"):
    """Load trained and initial models from checkpoint."""
    config_path = exp_dir / "training" / "config.yaml"
    with open(config_path) as f:
        config = yaml.safe_load(f)

    train_cfg = config["training"]
    net_config = NetworkConfig.from_dict(train_cfg)

    def make_model():
        return PDEOperatorNetwork(net_config)

    trained = make_model()
    ckpt_name = "final_model.pt" if (exp_dir / "checkpoints" / "final_model.pt").exists() else "best_model.pt"
    ckpt = torch.load(exp_dir / "checkpoints" / ckpt_name,
                       map_location=device, weights_only=False)
    trained.load_state_dict(ckpt["model_state_dict"])
    trained.to(device).eval()

    initial = make_model()
    ckpt0 = torch.load(exp_dir / "checkpoints" / "initial_model.pt",
                        map_location=device, weights_only=False)
    initial.load_state_dict(ckpt0["model_state_dict"])
    initial.to(device).eval()

    return trained, initial, config


def main():
    if len(sys.argv) < 2:
        print("Usage: uv run scripts/test_zero_step.py <experiment_dir>")
        sys.exit(1)

    exp_dir = Path(sys.argv[1])
    device = "cuda" if torch.cuda.is_available() else "cpu"

    trained, initial, config = load_models(exp_dir, device)
    pde_type = config["experiment"]["pde_type"]
    test_dir = Path(config["data"]["meta_test_dir"])
    k_shot = config["training"]["k_shot"]
    query_size = config["evaluation"].get("holdout_size", 5000)

    print(f"Model: {exp_dir.name}")
    print(f"PDE: {pde_type}, k_shot={k_shot}, query_size={query_size}")

    # Load test tasks via TASK_REGISTRY
    TaskClass = TASK_REGISTRY[pde_type]
    npz_files = sorted(test_dir.glob("*_fourier.npz"))
    tasks = [TaskClass(f, device=device) for f in npz_files]

    # E[alpha]
    alphas = [t.D for t in tasks]
    mean_alpha = np.mean(alphas)
    print(f"E[α] = {mean_alpha:.4f} ({len(tasks)} tasks, range [{min(alphas):.3f}, {max(alphas):.3f}])")

    # Pick one task per IC type
    ic_seen = {}
    selected = []
    for t in tasks:
        prefix = t.task_name.rsplit("_t", 1)[0] if "_t" in t.task_name else t.task_name.rsplit("_", 1)[0]
        if prefix not in ic_seen:
            ic_seen[prefix] = True
            selected.append(t)

    n_tasks = len(selected)
    print(f"Selected {n_tasks} tasks (one per IC type)")

    # Evaluate
    fig, axes = plt.subplots(n_tasks, 2, figsize=(12, 5 * n_tasks))
    if n_tasks == 1:
        axes = axes[np.newaxis, :]

    for i, task in enumerate(selected):
        # Get query split
        (_, _), (q_feats, q_tgts), _, _ = task.get_support_query_split(
            K_shot=k_shot, query_size=query_size, seed=42
        )

        true_ut = q_tgts[:, 0].cpu().numpy()

        # 0-step predictions
        with torch.no_grad():
            pred_trained = trained(q_feats).cpu().numpy()[:, 0]
            pred_initial = initial(q_feats).cpu().numpy()[:, 0]

        # Expected: E[α] * (u_xx + u_yy)
        laplacian = (q_feats[:, 3] + q_feats[:, 4]).cpu().numpy()
        pred_expected = mean_alpha * laplacian

        # Scatter: trained θ*
        ax = axes[i, 0]
        lim = max(abs(true_ut.max()), abs(true_ut.min())) * 1.1
        ax.scatter(true_ut, pred_trained, s=1, alpha=0.3, c="blue", label="θ* pred")
        ax.scatter(true_ut, pred_expected, s=1, alpha=0.15, c="green", label=f"E[α]·∇²u")
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, label="y=x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("True u_t")
        ax.set_ylabel("Pred u_t")
        ax.set_title(f"Trained θ* — {task.task_name} (D={task.D:.3f})")
        ax.legend(fontsize=8, markerscale=5)
        ax.set_aspect("equal")

        # Scatter: initial θ₀
        ax = axes[i, 1]
        ax.scatter(true_ut, pred_initial, s=1, alpha=0.3, c="gray", label="θ₀ pred")
        ax.scatter(true_ut, pred_expected, s=1, alpha=0.15, c="green", label=f"E[α]·∇²u")
        ax.plot([-lim, lim], [-lim, lim], "k--", alpha=0.5, label="y=x")
        ax.set_xlim(-lim, lim)
        ax.set_ylim(-lim, lim)
        ax.set_xlabel("True u_t")
        ax.set_ylabel("Pred u_t")
        ax.set_title(f"Initial θ₀ — {task.task_name} (D={task.D:.3f})")
        ax.legend(fontsize=8, markerscale=5)
        ax.set_aspect("equal")

        # Print correlation stats
        r_trained = np.corrcoef(true_ut, pred_trained)[0, 1]
        r_initial = np.corrcoef(true_ut, pred_initial)[0, 1]
        r_expected = np.corrcoef(true_ut, pred_expected)[0, 1]
        print(f"  {task.task_name} D={task.D:.3f}: "
              f"r(θ*)={r_trained:.4f}, r(θ₀)={r_initial:.4f}, r(E[α]·∇²u)={r_expected:.4f}")

    fig.suptitle(f"0-step predictions — {exp_dir.name}", fontsize=14)
    fig.tight_layout()
    out_path = exp_dir / "zero_step_scatter.png"
    fig.savefig(out_path, dpi=150)
    print(f"\nSaved: {out_path}")
    plt.close(fig)


if __name__ == "__main__":
    main()
