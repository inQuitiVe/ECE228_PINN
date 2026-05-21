import argparse
import os
import pickle

import matplotlib.pyplot as plt
import torch


def parse_args():
    parser = argparse.ArgumentParser(description="Plot steady loss history from a checkpoint or pickle file.")
    parser.add_argument(
        "--input",
        default="results/steady/checkpoints/steady_new.pt",
        help="Path to a .pt checkpoint or .pkl history file.",
    )
    parser.add_argument(
        "--output",
        default="results/steady/figures/steady_loss_curve.png",
        help="Output image path.",
    )
    return parser.parse_args()


def load_history(path):
    if path.endswith(".pt"):
        checkpoint = torch.load(path, map_location="cpu")
        return checkpoint.get("history", [])
    with open(path, "rb") as handle:
        return pickle.load(handle)


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

    os.makedirs(os.path.dirname(os.path.abspath(args.output)), exist_ok=True)
    plt.figure(figsize=(8, 4.5))
    plt.plot(iters, losses, linewidth=1.0)
    plt.xlabel("Iteration")
    plt.ylabel("Total Loss")
    plt.title("Steady Training Loss")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(args.output, dpi=180)


if __name__ == "__main__":
    main()
