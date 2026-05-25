"""
Evaluation harness for the unsteady PINN checkpoint.

Usage:
    python bench_unsteady.py \
        --checkpoint results/unsteady/checkpoints/latest.pt \
        --reference  data/reference/unsteady_reference.mat \
        --output-dir results/unsteady/bench
"""

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
import torch

from unsteady import PINNLaminarFlowTransient, load_checkpoint


# Probe locations matching paper Fig 8
PROBES = {
    "P1": (0.15, 0.20),
    "P2": (0.20, 0.25),
    "P3": (0.25, 0.20),
}


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate unsteady PINN checkpoint vs reference CFD")
    parser.add_argument("--checkpoint", default="results/unsteady/checkpoints/latest.pt")
    parser.add_argument("--reference",  default="data/reference/unsteady_reference.mat")
    parser.add_argument("--output-dir", default="results/unsteady/bench")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--tmax", type=float, default=0.5)
    parser.add_argument("--field-times", type=float, nargs="+", default=[0.3, 0.4, 0.5],
                        help="Snapshot times for side-by-side field comparison (paper Fig 7)")
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def l2_relative(pred, ref):
    """L2 relative error: ||pred - ref|| / ||ref||"""
    return np.linalg.norm(pred - ref) / (np.linalg.norm(ref) + 1e-12)


def demean(arr):
    return arr - arr.mean()


def nearest_snap(t_ref, t_query):
    """Return index of snapshot in t_ref closest to t_query."""
    return int(np.argmin(np.abs(t_ref - t_query)))


def plot_field_comparison(x, y, u_ref, v_ref, p_ref, u_pred, v_pred, p_pred,
                          t_val, out_path):
    """Side-by-side reference vs PINN for u, v, p at one time snapshot."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    kw = dict(alpha=0.7, edgecolors="none", cmap="rainbow", marker="o", s=1)
    pairs = [
        ("u ref", u_ref, "u PINN", u_pred, 0, 1.4),
        ("v ref", v_ref, "v PINN", v_pred, -0.7, 0.7),
        ("p ref (demeaned)", demean(p_ref), "p PINN (demeaned)", demean(p_pred), None, None),
    ]
    for row, (lbl_l, data_l, lbl_r, data_r, vmin, vmax) in enumerate(pairs):
        vmin_ = vmin if vmin is not None else np.min([data_l.min(), data_r.min()])
        vmax_ = vmax if vmax is not None else np.max([data_l.max(), data_r.max()])
        for col, (lbl, data) in enumerate([(lbl_l, data_l), (lbl_r, data_r)]):
            ax = axes[row, col]
            cf = ax.scatter(x, y, c=data, vmin=vmin_, vmax=vmax_, **kw)
            ax.set_title(f"{lbl}  t={t_val:.2f}s")
            ax.axis("square")
            ax.set_xlim([0, 1.1])
            ax.set_ylim([0, 0.41])
            fig.colorbar(cf, ax=ax, fraction=0.046, pad=0.04)
    plt.suptitle(f"Reference vs PINN   t = {t_val:.2f} s", fontsize=14)
    plt.tight_layout()
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    os.makedirs(args.output_dir, exist_ok=True)

    # ── load reference ────────────────────────────────────────────────────────
    mat = scipy.io.loadmat(args.reference)
    x_ref = mat["x"].flatten()        # (Ns,)
    y_ref = mat["y"].flatten()        # (Ns,)
    t_ref = mat["t"].flatten()        # (Nt,)
    U_ref = mat["u"]                  # (Ns, Nt)
    V_ref = mat["v"]                  # (Ns, Nt)
    P_ref = mat["p"]                  # (Ns, Nt)  — stores φ (instantaneous Chorin correction)
    Ns, Nt = U_ref.shape

    # ── load model ────────────────────────────────────────────────────────────
    uv_layers = [3] + 7 * [50] + [5]
    lb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ub = np.array([1.1, 0.41, args.tmax], dtype=np.float32)
    model = PINNLaminarFlowTransient(uv_layers=uv_layers, lb=lb, ub=ub).to(device)
    ckpt = load_checkpoint(model, args.checkpoint, map_location=device)
    model.eval()

    # ── per-snapshot L2 errors ────────────────────────────────────────────────
    l2_u = np.zeros(Nt)
    l2_v = np.zeros(Nt)
    l2_p = np.zeros(Nt)

    for k, t_val in enumerate(t_ref):
        t_query = np.full((Ns, 1), t_val, dtype=np.float32)
        u_pred, v_pred, p_pred = model.predict(x_ref, y_ref, t_query)
        u_pred = u_pred.flatten()
        v_pred = v_pred.flatten()
        p_pred = p_pred.flatten()

        l2_u[k] = l2_relative(u_pred, U_ref[:, k])
        l2_v[k] = l2_relative(v_pred, V_ref[:, k])
        # demeaned pressure comparison (Chorin φ vs PINN p)
        l2_p[k] = l2_relative(demean(p_pred), demean(P_ref[:, k]))

    mean_l2_u = float(np.mean(l2_u))
    mean_l2_v = float(np.mean(l2_v))
    mean_l2_p = float(np.mean(l2_p))

    print("=" * 50)
    print(f"Mean L2 u : {mean_l2_u*100:.2f}%")
    print(f"Mean L2 v : {mean_l2_v*100:.2f}%")
    print(f"Mean L2 p : {mean_l2_p*100:.2f}%  (demeaned)")
    go_u = mean_l2_u <= 0.10
    go_v = mean_l2_v <= 0.10
    print(f"Go/No-go  : L2_u {'PASS' if go_u else 'FAIL'}  L2_v {'PASS' if go_v else 'FAIL'}")
    print("=" * 50, flush=True)

    # ── L2 vs time plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_ref, l2_u * 100, label="L2_u (%)")
    ax.plot(t_ref, l2_v * 100, label="L2_v (%)")
    ax.plot(t_ref, l2_p * 100, label="L2_p demeaned (%)", linestyle="--")
    ax.axhline(10, color="red", linestyle=":", linewidth=1, label="10% threshold")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("L2 relative error (%)")
    ax.set_title(f"Per-snapshot L2 error   mean u={mean_l2_u*100:.1f}%  v={mean_l2_v*100:.1f}%")
    ax.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "l2_vs_time.png"), dpi=150)
    plt.close(fig)

    # ── probe pressure histories ───────────────────────────────────────────────
    fig, axes = plt.subplots(len(PROBES), 1, figsize=(8, 4 * len(PROBES)), sharex=True)
    for ax, (name, (xp, yp)) in zip(axes, PROBES.items()):
        # reference: find nearest fluid point
        dist = np.sqrt((x_ref - xp) ** 2 + (y_ref - yp) ** 2)
        idx = int(np.argmin(dist))
        p_probe_ref = P_ref[idx, :]

        # PINN prediction at probe over all time snapshots
        t_dense = t_ref
        xp_arr = np.full_like(t_dense, xp)
        yp_arr = np.full_like(t_dense, yp)
        _, _, p_probe_pred = model.predict(xp_arr, yp_arr, t_dense.reshape(-1, 1))
        p_probe_pred = p_probe_pred.flatten()

        ax.plot(t_ref, demean(p_probe_ref), label="Reference (demeaned)")
        ax.plot(t_ref, demean(p_probe_pred), label="PINN (demeaned)", linestyle="--")
        ax.set_ylabel("p (Pa)")
        ax.set_title(f"Probe {name} at ({xp}, {yp})")
        ax.legend()
    axes[-1].set_xlabel("t (s)")
    plt.suptitle("Probe pressure histories (Fig 8 comparison)", fontsize=13)
    plt.tight_layout()
    plt.savefig(os.path.join(args.output_dir, "probe_pressures.png"), dpi=150)
    plt.close(fig)

    # ── field comparison at selected snapshots (paper Fig 7) ─────────────────
    for t_query in args.field_times:
        k = nearest_snap(t_ref, t_query)
        t_val = t_ref[k]
        t_arr = np.full((Ns, 1), t_val, dtype=np.float32)
        u_pred, v_pred, p_pred = model.predict(x_ref, y_ref, t_arr)
        u_pred = u_pred.flatten()
        v_pred = v_pred.flatten()
        p_pred = p_pred.flatten()
        fname = f"field_comparison_t{t_val:.2f}.png".replace(".", "p")
        plot_field_comparison(
            x_ref, y_ref,
            U_ref[:, k], V_ref[:, k], P_ref[:, k],
            u_pred, v_pred, p_pred,
            t_val,
            os.path.join(args.output_dir, fname),
        )
        print(f"Saved field comparison at t={t_val:.3f}s → {fname}", flush=True)

    print(f"\nAll outputs written to {args.output_dir}", flush=True)


if __name__ == "__main__":
    main()
