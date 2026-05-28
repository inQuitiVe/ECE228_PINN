"""
Evaluation harness for the unsteady PINN checkpoint.

Usage:
    python bench_unsteady.py \
        --checkpoint results/experiments/vanilla/checkpoints/latest.pt \
        --reference  data/reference/unsteady_reference.mat \
        --output-dir results/experiments/vanilla/benchmarks/latest_lbfgs
"""

import argparse
import csv
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
    parser.add_argument("--exp-name", default="vanilla",
                        help="Experiment name; derives default checkpoint and output-dir paths.")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--reference",  default="data/reference/unsteady_reference.mat")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--tmax", type=float, default=0.5)
    parser.add_argument("--field-times", type=float, nargs="+", default=[0.3, 0.4, 0.5],
                        help="Snapshot times for side-by-side field comparison (paper Fig 7)")
    parser.add_argument("--t-developed", type=float, default=0.1,
                        help="Only snapshots with t >= this value count toward the primary L2 metric")
    parser.add_argument("--diagnostic-times", type=float, nargs="+",
                        default=[0.1, 0.2, 0.3, 0.4, 0.5],
                        help="Snapshot times for detailed v-field metric diagnostics")
    args = parser.parse_args()
    run_dir = f"results/experiments/{args.exp_name}"
    if not args.checkpoint:
        args.checkpoint = f"{run_dir}/checkpoints/latest_lbfgs.pt"
        if not os.path.exists(args.checkpoint):
            args.checkpoint = f"{run_dir}/checkpoints/best.pt"
    if not args.output_dir:
        args.output_dir = f"{run_dir}/benchmarks/latest_lbfgs"
    return args


def resolve_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def l2_relative(pred, ref, min_norm=1e-2):
    """L2 relative error: ||pred - ref|| / ||ref||.
    Returns NaN when ||ref|| < min_norm to avoid division by near-zero
    (early snapshots where inlet is closed and the flow is at rest)."""
    norm_ref = np.linalg.norm(ref)
    if norm_ref < min_norm:
        return np.nan
    return np.linalg.norm(pred - ref) / norm_ref


def l2_global(pred_all, ref_all):
    """Global (space+time) Frobenius L2: ||P-R||_F / ||R||_F.
    Primary scalar metric — unaffected by near-zero early snapshots."""
    return np.linalg.norm(pred_all - ref_all) / (np.linalg.norm(ref_all) + 1e-12)


def demean(arr):
    return arr - arr.mean()


def field_diagnostics(pred, ref, min_norm=1e-2):
    norm_ref = np.linalg.norm(ref)
    norm_pred = np.linalg.norm(pred)
    if norm_ref < min_norm:
        l2 = np.nan
    else:
        l2 = np.linalg.norm(pred - ref) / norm_ref
    if norm_ref < 1e-12 or norm_pred < 1e-12:
        cosine = np.nan
    else:
        cosine = float(np.dot(pred, ref) / (norm_pred * norm_ref))
    return {
        "ref_norm": norm_ref,
        "pred_norm": norm_pred,
        "ref_rms": float(np.sqrt(np.mean(ref ** 2))),
        "pred_rms": float(np.sqrt(np.mean(pred ** 2))),
        "ref_max_abs": float(np.max(np.abs(ref))),
        "pred_max_abs": float(np.max(np.abs(pred))),
        "pred_ref_norm_ratio": float(norm_pred / norm_ref) if norm_ref > 1e-12 else np.nan,
        "cosine": cosine,
        "l2": l2,
        "near_zero_ref": bool(norm_ref < min_norm),
    }


def nearest_snap(t_ref, t_query):
    """Return index of snapshot in t_ref closest to t_query."""
    return int(np.argmin(np.abs(t_ref - t_query)))


def plot_field_comparison(x, y, u_ref, v_ref, p_ref, u_pred, v_pred, p_pred,
                          t_val, out_path):
    """Side-by-side reference vs PINN for u, v, p at one time snapshot."""
    fig, axes = plt.subplots(3, 2, figsize=(12, 8))
    kw = dict(alpha=0.7, edgecolors="none", cmap="rainbow", marker="o", s=1)
    pairs = [
        ("u ref", u_ref, "u PINN", u_pred, None, None),
        ("v ref", v_ref, "v PINN", v_pred, None, None),
        ("p ref (demeaned)", demean(p_ref), "p PINN (demeaned)", demean(p_pred), None, None),
    ]
    for row, (lbl_l, data_l, lbl_r, data_r, vmin, vmax) in enumerate(pairs):
        vmin_ = vmin if vmin is not None else np.nanmin([np.nanmin(data_l), np.nanmin(data_r)])
        vmax_ = vmax if vmax is not None else np.nanmax([np.nanmax(data_l), np.nanmax(data_r)])
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
    l2_u = np.full(Nt, np.nan)
    l2_v = np.full(Nt, np.nan)
    l2_p = np.full(Nt, np.nan)
    U_pred_all = np.zeros((Ns, Nt), dtype=np.float32)
    V_pred_all = np.zeros((Ns, Nt), dtype=np.float32)
    P_pred_all = np.zeros((Ns, Nt), dtype=np.float32)

    for k, t_val in enumerate(t_ref):
        t_query = np.full((Ns, 1), t_val, dtype=np.float32)
        u_pred, v_pred, p_pred = model.predict(x_ref, y_ref, t_query)
        u_pred = u_pred.flatten()
        v_pred = v_pred.flatten()
        p_pred = p_pred.flatten()
        U_pred_all[:, k] = u_pred
        V_pred_all[:, k] = v_pred
        P_pred_all[:, k] = p_pred

        l2_u[k] = l2_relative(u_pred, U_ref[:, k])
        l2_v[k] = l2_relative(v_pred, V_ref[:, k])
        l2_p[k] = l2_relative(demean(p_pred), demean(P_ref[:, k]))

    # Global Frobenius norm (primary metric — robust to near-zero early snapshots)
    global_l2_u = l2_global(U_pred_all, U_ref)
    global_l2_v = l2_global(V_pred_all, V_ref)
    global_l2_p = l2_global(
        P_pred_all - P_pred_all.mean(axis=1, keepdims=True),
        P_ref      - P_ref.mean(axis=1, keepdims=True)
    )
    # Primary metric: developed-flow snapshots only (t >= t_developed)
    dev_mask = t_ref >= args.t_developed
    dev_l2_u = float(np.nanmean(l2_u[dev_mask]))
    dev_l2_v = float(np.nanmean(l2_v[dev_mask]))
    dev_l2_p = float(np.nanmean(l2_p[dev_mask]))
    n_dev = int(np.sum(dev_mask & ~np.isnan(l2_u)))
    n_valid = int(np.sum(~np.isnan(l2_u)))

    print("=" * 50)
    print(f"PRIMARY METRIC (t >= {args.t_developed}s, developed flow, {n_dev} snapshots):")
    print(f"  Mean L2 u : {dev_l2_u*100:.2f}%")
    print(f"  Mean L2 v : {dev_l2_v*100:.2f}%")
    print(f"  Mean L2 p : {dev_l2_p*100:.2f}%  (demeaned)")
    go_u = dev_l2_u <= 0.10
    go_v = dev_l2_v <= 0.10
    print(f"  Go/No-go  : L2_u {'PASS' if go_u else 'FAIL'}  L2_v {'PASS' if go_v else 'FAIL'}  (≤ 10%)")
    print(f"REFERENCE (all {n_valid}/{Nt} non-NaN snapshots):")
    print(f"  Global Frobenius L2 u : {global_l2_u*100:.2f}%")
    print(f"  Global Frobenius L2 v : {global_l2_v*100:.2f}%")
    print(f"  Global Frobenius L2 p : {global_l2_p*100:.2f}%  (demeaned)")
    print("=" * 50, flush=True)

    v_diagnostics = []
    for k, t_val in enumerate(t_ref):
        diag = field_diagnostics(V_pred_all[:, k], V_ref[:, k])
        diag["t"] = float(t_val)
        v_diagnostics.append(diag)

    dev_v = [diag for diag in v_diagnostics if diag["t"] >= args.t_developed and not diag["near_zero_ref"]]
    if dev_v:
        ratio_mean = float(np.nanmean([diag["pred_ref_norm_ratio"] for diag in dev_v]))
        cosine_mean = float(np.nanmean([diag["cosine"] for diag in dev_v]))
        ref_rms_min = float(np.nanmin([diag["ref_rms"] for diag in dev_v]))
        ref_rms_max = float(np.nanmax([diag["ref_rms"] for diag in dev_v]))
        pred_rms_mean = float(np.nanmean([diag["pred_rms"] for diag in dev_v]))
        print("V-FIELD DIAGNOSTICS (t >= %.3fs):" % args.t_developed)
        print("  ref_rms range       : %.3e .. %.3e" % (ref_rms_min, ref_rms_max))
        print("  pred_rms mean       : %.3e" % pred_rms_mean)
        print("  mean |pred|/|ref|   : %.3f" % ratio_mean)
        print("  mean cosine(pred,ref): %.3f" % cosine_mean)
        print("  Interpretation hint : L2≈100%% with |pred|/|ref|≈0 means near-zero prediction;")
        print("                        L2≈100%% with nonzero |pred|/|ref| and low cosine means spatial/phase mismatch.")
        print("  Selected snapshots:")
        print("    t      ref_rms   pred_rms  |pred|/|ref|  cosine  L2_v     note")
        for t_query in args.diagnostic_times:
            k = nearest_snap(t_ref, t_query)
            diag = v_diagnostics[k]
            note = "near-zero ref" if diag["near_zero_ref"] else ""
            print(
                "    %.2f  %.3e  %.3e     %.3f      %.3f  %6.1f%%  %s"
                % (
                    diag["t"],
                    diag["ref_rms"],
                    diag["pred_rms"],
                    diag["pred_ref_norm_ratio"],
                    diag["cosine"],
                    diag["l2"] * 100 if np.isfinite(diag["l2"]) else np.nan,
                    note,
                )
            )
        print("=" * 50, flush=True)

    diag_csv = os.path.join(args.output_dir, "v_diagnostics.csv")
    with open(diag_csv, "w", newline="") as handle:
        writer = csv.DictWriter(
            handle,
            fieldnames=[
                "t",
                "ref_norm",
                "pred_norm",
                "ref_rms",
                "pred_rms",
                "ref_max_abs",
                "pred_max_abs",
                "pred_ref_norm_ratio",
                "cosine",
                "l2",
                "near_zero_ref",
            ],
        )
        writer.writeheader()
        writer.writerows(v_diagnostics)
    print(f"Saved v-field diagnostics → {diag_csv}", flush=True)

    # ── per-snapshot L2 CSV (u, v, p) ────────────────────────────────────────
    snap_csv = os.path.join(args.output_dir, "snapshot_metrics.csv")
    with open(snap_csv, "w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["t", "l2_u", "l2_v", "l2_p"])
        writer.writeheader()
        for k, t_val in enumerate(t_ref):
            writer.writerow({
                "t": float(t_val),
                "l2_u": float(l2_u[k]) if np.isfinite(l2_u[k]) else "",
                "l2_v": float(l2_v[k]) if np.isfinite(l2_v[k]) else "",
                "l2_p": float(l2_p[k]) if np.isfinite(l2_p[k]) else "",
            })
    print(f"Saved per-snapshot L2 metrics → {snap_csv}", flush=True)

    # ── L2 vs time plot ───────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.plot(t_ref, l2_u * 100, label="L2_u (%)")
    ax.plot(t_ref, l2_v * 100, label="L2_v (%)")
    ax.plot(t_ref, l2_p * 100, label="L2_p demeaned (%)", linestyle="--")
    ax.axhline(10, color="red", linestyle=":", linewidth=1, label="10% threshold")
    ax.axvline(args.t_developed, color="gray", linestyle="--", linewidth=1,
               label=f"t={args.t_developed}s (developed)")
    ax.set_xlabel("t (s)")
    ax.set_ylabel("L2 relative error (%)")
    ax.set_title(f"Per-snapshot L2   u={dev_l2_u*100:.1f}%  v={dev_l2_v*100:.1f}%  (t≥{args.t_developed}s)")
    ax.set_ylim(bottom=0)
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
        t_str = f"{t_val:.2f}".replace(".", "p")
        fname = f"field_comparison_t{t_str}.png"
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
