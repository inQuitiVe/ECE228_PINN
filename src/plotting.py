import argparse
import os
import pickle

import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Plot steady loss history from a checkpoint or pickle file.")
    parser.add_argument(
        "--input",
        default="results/steady/checkpoints/latest.pt",
        help="Path to a .pt checkpoint or .pkl history file.",
    )
    parser.add_argument(
        "--output",
        default="results/steady/figures/loss_curve.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--manual-fix-start",
        type=int,
        default=0,
        help="Start iteration for manual interpolation fix (inclusive). Disabled when 0.",
    )
    parser.add_argument(
        "--manual-fix-end",
        type=int,
        default=0,
        help="End iteration for manual interpolation fix (inclusive). Disabled when 0.",
    )
    return parser.parse_args()


def load_history(path):
    if path.endswith(".pt"):
        checkpoint = torch.load(path, map_location="cpu")
        return checkpoint.get("history", [])
    with open(path, "rb") as handle:
        return pickle.load(handle)


def plot_loss(history, output_path):
    iters = []
    losses = []
    for idx, row in enumerate(history, start=1):
        if "loss" not in row:
            continue
        iters.append(int(row.get("iter", idx)))
        losses.append(float(row["loss"]))

    output_dir = os.path.dirname(os.path.abspath(output_path))
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(iters, losses, linewidth=1.0)
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("Steady Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(output_path, dpi=180)


def apply_manual_interpolation(iters, losses, start_iter, end_iter):
    if start_iter <= 0 or end_iter <= 0 or end_iter <= start_iter:
        return losses
    if start_iter not in iters or end_iter not in iters:
        return losses
    start_idx = iters.index(start_iter)
    end_idx = iters.index(end_iter)
    if end_idx - start_idx < 2:
        return losses

    left_value = losses[start_idx]
    right_value = losses[end_idx]
    span = end_idx - start_idx
    fixed = list(losses)
    for idx in range(start_idx + 1, end_idx):
        alpha = (idx - start_idx) / span
        fixed[idx] = (1.0 - alpha) * left_value + alpha * right_value
    return fixed


def main():
    args = parse_args()
    history = load_history(args.input)
    iters = []
    losses = []
    for idx, row in enumerate(history, start=1):
        if "loss" not in row:
            continue
        iters.append(int(row.get("iter", idx)))
        losses.append(float(row["loss"]))

    losses = apply_manual_interpolation(
        iters=iters,
        losses=losses,
        start_iter=args.manual_fix_start,
        end_iter=args.manual_fix_end,
    )
    plot_loss(
        history=[{"iter": it, "loss": lo} for it, lo in zip(iters, losses)],
        output_path=args.output,
    )


if __name__ == "__main__":
    main()
