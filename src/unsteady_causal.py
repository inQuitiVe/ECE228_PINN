"""Causal training for the transient mixed-form PINN.

Builds on unsteady.py (vanilla) by replacing the PDE-residual term `loss_f`
with a causally-weighted version following

    Wang, Sankaran & Perdikaris,
    "Respecting Causality for Training Physics-Informed Neural Networks",
    CMAME 2024 (arXiv:2203.07404).

Causal weight (stop-gradient), with two deliberate refinements over the bare
paper formula (both flagged in the CLI):

    L_i    = mean mixed-form residual in temporal bin i      (solution quality)
    w_i    = exp( -(eps / M) * sum_{k<i} L_k )               (causal gate)
    loss_f = (1/N) * sum_i  w_i * sum_{x in bin i} R(x)      (count-weighted)

Refinement 1 — count weighting (`--causal-bin-mode`):
    Bins can be equal physical DURATION (default; robust to non-uniform /
    adaptive sampling) or equal COUNT (ablation). The (1/N)·sum form weights
    each bin by its point count N_i implicitly, so loss_f reduces EXACTLY to
    the vanilla mean residual at eps=0 under BOTH binnings, for any point
    distribution.

Refinement 2 — eps/M normalization:
    Dividing the exponent by M makes eps's decay scale independent of the
    number of bins, so changing M does not require re-tuning eps. (This is a
    deliberate deviation from the bare paper formula; note in the report.)

Refinement 3 — L-BFGS weight handling (`--causal-lbfgs`):
    'uniform' (default) optimizes the true unweighted loss; 'frozen' continues
    the Adam-final causal weights as STATIC constants (L-BFGS-safe); 'dynamic'
    recomputes weights each closure (breaks the quasi-Newton static-objective
    assumption — kept only for experimentation, warns).

Only loss_f is modified; BC/IC terms and their beta weights (5/5/1/1) are
identical to unsteady.py build_loss.
"""

import argparse
import os
import pickle
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

from unsteady import (
    PINNLaminarFlowTransient,
    build_training_data,
    load_checkpoint,
    mse_zero,
    post_process,
    resolve_device,
    save_checkpoint,
)


def _fmt_hms(seconds):
    """Compact h/m/s for progress + ETA lines."""
    seconds = int(max(0, seconds))
    h, rem = divmod(seconds, 3600)
    m, s = divmod(rem, 60)
    if h:
        return f"{h}h{m:02d}m{s:02d}s"
    return f"{m}m{s:02d}s"


def compute_bins(t_c, M, tmax, mode, device):
    """Assign each collocation point to one of M temporal bins.

    mode='duration': equal physical-time slices of width tmax/M (default).
        Robust to non-uniform sampling; bin i covers t in [i·tmax/M, (i+1)·tmax/M).
        Counts N_i may vary.
    mode='count': sort by t and chunk into M equal-count groups (N_i equal;
        bin widths vary with point density).

    Returns (bin_idx, safe_cnt, n_total). The count-weighted loss reduces
    EXACTLY to vanilla mean residual at eps=0 under both modes.
    """
    t_flat = t_c.detach().reshape(-1)
    n = t_flat.numel()
    if mode == "duration":
        bin_idx = torch.clamp((t_flat / tmax * M).long(), 0, M - 1)
    elif mode == "count":
        order = torch.argsort(t_flat)
        ranks = torch.empty(n, dtype=torch.long, device=device)
        ranks[order] = torch.arange(n, device=device)
        bin_idx = torch.clamp((ranks * M) // n, max=M - 1)
    else:
        raise ValueError(f"unknown bin mode {mode!r}")
    cnt = torch.zeros(M, device=device).scatter_add(0, bin_idx, torch.ones(n, device=device))
    safe_cnt = cnt.clamp(min=1.0)
    return bin_idx, safe_cnt, n


def causal_loss_f(model, data, binspec, eps, fixed_w=None):
    """Causally-weighted PDE residual (count-weighted, eps/M-normalized).

    binspec = (bin_idx, safe_cnt, n_total, M).
    fixed_w (length-M tensor) overrides the dynamic weights — used to FREEZE
    the Adam-final weights during L-BFGS so the objective stays static, and to
    supply all-ones for the 'uniform' (true-loss) L-BFGS mode.
    Returns (loss_f_weighted, loss_f_true, per_bin_loss[detached], weights[detached]).
    loss_f_true is the unweighted mean residual (eps=0 equivalent), for
    monitoring early-stop / the LR scheduler on a stationary signal.
    """
    bin_idx, safe_cnt, n_total, M = binspec
    f_u, f_v, f_s11, f_s22, f_s12, f_p = model.net_f(data["x_c"], data["y_c"], data["t_c"])
    r = (f_u.square() + f_v.square() + f_s11.square()
         + f_s22.square() + f_s12.square() + f_p.square()).reshape(-1)
    sum_per_bin = torch.zeros(M, device=r.device).scatter_add(0, bin_idx, r)
    per_bin_loss = sum_per_bin / safe_cnt
    if fixed_w is not None:
        weights = fixed_w
    else:
        with torch.no_grad():
            cum_prev = torch.cumsum(per_bin_loss, dim=0) - per_bin_loss
            weights = torch.exp(-(eps / M) * cum_prev)
    loss_f = (weights * sum_per_bin).sum() / n_total
    loss_f_true = (sum_per_bin.sum() / n_total).detach()
    return loss_f, loss_f_true, per_bin_loss.detach(), weights.detach()


def build_causal_loss(model, data, binspec, eps, fixed_w=None):
    """build_loss clone whose loss_f is causally weighted (beta weights unchanged).

    Returns both the weighted "loss" (backprop target) and the unweighted
    "loss_true" (= loss with the eps=0 residual), which early-stop and the LR
    scheduler monitor as a stationary convergence signal.
    """
    loss_f, loss_f_true, per_bin_loss, weights = causal_loss_f(model, data, binspec, eps, fixed_w)

    u_ic, v_ic, p_ic, _, _, _ = model.net_uv(data["x_ic"], data["y_ic"], data["t_ic"])
    u_wall, v_wall, _, _, _, _ = model.net_uv(data["x_wall"], data["y_wall"], data["t_wall"])
    u_inlet, v_inlet, _, _, _, _ = model.net_uv(data["x_inlet"], data["y_inlet"], data["t_inlet"])
    _, _, p_outlet, _, _, _ = model.net_uv(data["x_outlet"], data["y_outlet"], data["t_outlet"])

    loss_ic = mse_zero(u_ic) + mse_zero(v_ic) + mse_zero(p_ic)
    loss_wall = mse_zero(u_wall) + mse_zero(v_wall)
    loss_inlet = (torch.mean((u_inlet - data["u_inlet"]).square())
                  + torch.mean((v_inlet - data["v_inlet"]).square()))
    loss_outlet = mse_zero(p_outlet)
    bc = 5.0 * loss_wall + 5.0 * loss_inlet + loss_outlet + loss_ic
    loss = loss_f + bc
    # Unweighted true loss: BC/IC terms identical, only loss_f swapped for the eps=0 residual.
    loss_true = float(loss_f_true) + float(bc.detach())
    return {
        "loss": loss,
        "loss_f": loss_f,
        "loss_f_true": float(loss_f_true),
        "loss_true": loss_true,
        "loss_ic": loss_ic,
        "loss_wall": loss_wall,
        "loss_inlet": loss_inlet,
        "loss_outlet": loss_outlet,
        "causal_weights": weights,
        "per_bin_loss": per_bin_loss,
        "min_w": float(weights.min()),
    }


def parse_args():
    parser = argparse.ArgumentParser(description="Causal training for transient mixed-form PINN")
    # ---- identical to unsteady.py ----
    parser.add_argument("--adam-iters", type=int, default=10000)
    parser.add_argument("--lbfgs-iters", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--tmax", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.005,
                        help="Dynamic viscosity (Re=10: 0.005, Re=100: 0.0005)")
    parser.add_argument("--period", type=float, default=1.0)
    parser.add_argument("--exp-name", default="causal")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--best-checkpoint", default="")
    parser.add_argument("--lbfgs-checkpoint", default="")
    parser.add_argument("--load-checkpoint", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--lbfgs-save-every", type=int, default=500)
    parser.add_argument("--snapshot-iters", type=str, default="",
                        help="Comma-separated Adam iters for non-overwriting snapshot_iter{N}.pt")
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--loss-history", default="")
    parser.add_argument("--num-frames", type=int, default=51)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--print-every", type=int, default=100)
    # ---- scheduler + early-stop (monitor the unweighted true loss) ----
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine", "step", "plateau"])
    parser.add_argument("--scheduler-step-size", type=int, default=1000)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
    parser.add_argument("--scheduler-plateau-patience", type=int, default=500)
    parser.add_argument("--early-stop-patience", type=int, default=0,
                        help="Stop after N stale steps on the true loss (0 = off). "
                             "Only fires once min_w > --causality-delta.")
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-warmup", type=int, default=0)
    # ---- causal-specific ----
    parser.add_argument("--causal-eps", type=float, default=1.0,
                        help="Causality strength eps (exponent normalized by M). "
                             "0 → reduces exactly to vanilla loss_f.")
    parser.add_argument("--causal-bins", type=int, default=32,
                        help="Number of temporal bins M.")
    parser.add_argument("--causal-bin-mode", default="duration", choices=["duration", "count"],
                        help="'duration' (equal physical-time slices, default) or "
                             "'count' (equal-count chunks).")
    parser.add_argument("--causal-lbfgs", default="uniform", choices=["uniform", "frozen", "dynamic"],
                        help="L-BFGS residual weighting: 'uniform' (true loss, default), "
                             "'frozen' (Adam-final weights as static constants, L-BFGS-safe), "
                             "'dynamic' (recompute each closure — may break line search).")
    parser.add_argument("--causality-delta", type=float, default=0.9,
                        help="Early-stop only allowed once min_w exceeds this "
                             "(Wang 2024 'causality satisfied' gate).")
    parser.add_argument("--freeze-weight-checkpoint", default="",
                        help="For --causal-lbfgs frozen: compute the frozen bin weights "
                             "from THIS checkpoint's model (e.g. the Adam-best / "
                             "snapshot_iter20000), while optimising from --load-checkpoint's "
                             "model. Decouples the freeze point from the L-BFGS start, so a "
                             "continued run keeps the ORIGINAL Adam-end frozen weights instead "
                             "of re-freezing at the already-converged state.")
    args = parser.parse_args()
    if args.causal_bins < 1:
        parser.error("--causal-bins must be >= 1")
    if args.causal_eps < 0:
        parser.error("--causal-eps must be >= 0")
    return args


def train_causal(
    model, data, binspec, eps,
    adam_iters, learning_rate, lbfgs_steps, print_every, lbfgs_mode,
    scheduler_type="none", scheduler_step_size=1000, scheduler_gamma=0.5,
    scheduler_min_lr=1e-6, scheduler_plateau_patience=500,
    early_stop_patience=0, early_stop_min_delta=0.0, early_stop_warmup=0,
    causality_delta=0.9,
    start_iter=1, checkpoint_path=None, save_every=0, snapshot_iters=None,
    save_best=False, best_checkpoint_path=None,
    lbfgs_checkpoint_path=None, lbfgs_save_every=0,
    optimizer_state_dict=None, scheduler_state_dict=None,
    existing_history=None, weight_history=None, initial_best_loss=None,
    freeze_weight_state=None,
):
    M = binspec[3]
    device = data["x_c"].device
    history = list(existing_history) if existing_history else []
    weight_history = list(weight_history) if weight_history else []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, adam_iters), eta_min=scheduler_min_lr)
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, scheduler_step_size), gamma=scheduler_gamma)
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_gamma,
            patience=max(1, scheduler_plateau_patience), min_lr=scheduler_min_lr)
    else:
        scheduler = None
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if scheduler is not None and scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    start_time = time.time()
    model.train()
    # Early-stop / best tracked on the UNWEIGHTED true loss (stationary signal).
    best_true = float(initial_best_loss) if initial_best_loss is not None else float("inf")
    best_record = {"loss": best_true, "iter": 0}
    stale_steps = 0
    causality_satisfied = False

    print("=== Adam phase | %d iters | lr=%.2e | sched=%s | early-stop: patience=%d "
          "warmup=%d gate(min_w>%.2f) ===" % (adam_iters, learning_rate, scheduler_type,
          early_stop_patience, early_stop_warmup, causality_delta), flush=True)

    for iteration in range(start_iter, adam_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = build_causal_loss(model, data, binspec, eps, fixed_w=None)
        losses["loss"].backward()
        optimizer.step()

        loss_true = losses["loss_true"]
        min_w = losses["min_w"]
        # The defining causal-training milestone: the curriculum has reached the
        # final temporal bin (all w_i > delta). Log once; early-stop arms here.
        if not causality_satisfied and min_w > causality_delta and iteration > 1:
            causality_satisfied = True
            print("[causality] SATISFIED @ Adam %d: min_w=%.3f > %.2f — all %d bins "
                  "unlocked; early-stop now armed." % (iteration, min_w, causality_delta, M),
                  flush=True)
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(loss_true)   # monitor the stationary true loss
            else:
                scheduler.step()

        snapshot = {k: losses[k].detach().cpu().item()
                    for k in ("loss", "loss_f", "loss_ic", "loss_wall", "loss_inlet", "loss_outlet")}
        snapshot["iter"] = iteration
        snapshot["min_w"] = min_w
        snapshot["loss_true"] = loss_true
        snapshot["loss_f_true"] = losses["loss_f_true"]
        snapshot["lr"] = optimizer.param_groups[0]["lr"]
        history.append(snapshot)

        if iteration == 1 or iteration % print_every == 0:
            w = losses["causal_weights"]
            unlocked = int((w > causality_delta).sum().item())
            elapsed = time.time() - start_time
            done = max(1, iteration - start_iter + 1)
            rate = done / elapsed if elapsed > 0 else 0.0
            eta = _fmt_hms((adam_iters - iteration) / rate) if rate > 0 else "?"
            weight_history.append({
                "iter": iteration,
                "weights": w.cpu().tolist(),
                "per_bin_loss": losses["per_bin_loss"].cpu().tolist(),
                "min_w": min_w,
            })
            print(
                "Adam %d/%d | total=%.3e true=%.3e f=%.3e ic=%.3e wall=%.3e inlet=%.3e outlet=%.3e "
                "| min_w=%.3e unlocked=%d/%d | best=%.3e@%d stale=%d | lr=%.2e | %.1f it/s ETA %s"
                % (iteration, adam_iters, snapshot["loss"], loss_true, snapshot["loss_f"],
                   snapshot["loss_ic"], snapshot["loss_wall"], snapshot["loss_inlet"],
                   snapshot["loss_outlet"], min_w, unlocked, M, best_true, best_record["iter"],
                   stale_steps, snapshot["lr"], rate, eta),
                flush=True,
            )

        # Best + early-stop on the true loss, gated by causality (min_w > delta).
        if loss_true < (best_true - early_stop_min_delta):
            best_true = loss_true
            stale_steps = 0
            best_record = {"loss": loss_true, "iter": iteration}
            if save_best and best_checkpoint_path:
                save_checkpoint(
                    model, best_checkpoint_path, history,
                    {"best_iter": iteration, "best_loss": loss_true},
                    optimizer=optimizer, scheduler=scheduler,
                    extra_state={"iteration": iteration, "best_loss": loss_true,
                                 "causal_eps": eps, "causal_bins": M},
                )
        elif iteration > early_stop_warmup:
            stale_steps += 1
            if (early_stop_patience > 0 and stale_steps >= early_stop_patience
                    and min_w > causality_delta):
                print("Early stop @ Adam %d (true-loss stale %d steps, causality satisfied "
                      "min_w=%.3f > %.2f)" % (iteration, stale_steps, min_w, causality_delta),
                      flush=True)
                break

        if save_every > 0 and checkpoint_path and iteration % save_every == 0:
            save_checkpoint(
                model, checkpoint_path, history, {"last_iter": iteration},
                optimizer=optimizer, scheduler=scheduler,
                extra_state={"iteration": iteration, "best_loss": best_true,
                             "causal_eps": eps, "causal_bins": M},
            )

        if snapshot_iters and iteration in snapshot_iters and checkpoint_path:
            snap_path = os.path.join(os.path.dirname(checkpoint_path),
                                     f"snapshot_iter{iteration}.pt")
            save_checkpoint(
                model, snap_path, history,
                {"snapshot_iter": iteration, "phase": "adam"},
                optimizer=optimizer, scheduler=scheduler,
                extra_state={"iteration": iteration, "best_loss": best_true,
                             "causal_eps": eps, "causal_bins": M},
            )
            print(f"[snapshot] iter {iteration} → {snap_path}", flush=True)

    # ---- L-BFGS phase ----
    if lbfgs_steps > 0:
        # Resolve a STATIC weight vector for the L-BFGS objective (Refinement 3).
        if lbfgs_mode == "uniform":
            lbfgs_fixed_w = torch.ones(M, device=device)
            print("[lbfgs] uniform: optimizing the true unweighted loss.", flush=True)
        elif lbfgs_mode == "frozen":
            # net_f needs autograd to derive u,v from psi, so this cannot run under
            # no_grad; the returned weights are already detached by causal_loss_f.
            if freeze_weight_state is not None:
                # Frozen weights are a deterministic function of the model state, so to
                # reproduce the ORIGINAL Adam-end weights on a continued run we temporarily
                # load the freeze-source model (e.g. Adam-best), read the weights off, then
                # restore the current (e.g. resumed latest_lbfgs) model to keep optimising.
                saved = {k: v.detach().clone() for k, v in model.state_dict().items()}
                model.load_state_dict(freeze_weight_state)
                _, _, _, w_end = causal_loss_f(model, data, binspec, eps, fixed_w=None)
                model.load_state_dict(saved)
                print("[lbfgs] frozen weights taken from --freeze-weight-checkpoint "
                      "(decoupled from the L-BFGS start).", flush=True)
            else:
                _, _, _, w_end = causal_loss_f(model, data, binspec, eps, fixed_w=None)
            lbfgs_fixed_w = w_end
            print("[lbfgs] frozen: continuing Adam-final causal weights as constants "
                  "(min_w=%.3e, max_w=%.3e)." % (float(w_end.min()), float(w_end.max())), flush=True)
        else:  # dynamic
            lbfgs_fixed_w = None
            print("[lbfgs] WARNING dynamic: recomputing causal weights each closure — "
                  "this violates L-BFGS's static-objective assumption and may cause "
                  "line-search failure / early termination.", flush=True)

        print("=== L-BFGS phase | max_iter=%d | mode=%s ===" % (lbfgs_steps, lbfgs_mode),
              flush=True)
        lbfgs_start = time.time()
        lbfgs_best = {"loss": float("inf"), "iter": 0}

        # Standard PINN L-BFGS: real tolerances, single step(), natural convergence.
        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0,
            max_iter=lbfgs_steps, max_eval=int(lbfgs_steps * 1.25),
            history_size=50,
            tolerance_grad=1e-7, tolerance_change=1.0 * np.finfo(float).eps,
            line_search_fn="strong_wolfe",
        )
        step_counter = {"count": 0}
        last_iter = max((int(h.get("iter", 0)) for h in history), default=start_iter - 1)

        def make_snapshot(losses_dict, lbfgs_iter):
            snap = {k: losses_dict[k].detach().cpu().item()
                    for k in ("loss", "loss_f", "loss_ic", "loss_wall", "loss_inlet", "loss_outlet")}
            snap["iter"] = last_iter + lbfgs_iter
            snap["phase"] = "lbfgs"
            snap["lbfgs_iter"] = lbfgs_iter
            snap["min_w"] = losses_dict["min_w"]
            snap["loss_true"] = losses_dict["loss_true"]
            snap["loss_f_true"] = losses_dict["loss_f_true"]
            return snap

        def save_lbfgs_ckpt(losses_dict, lbfgs_iter, final=False):
            if not lbfgs_checkpoint_path:
                return
            snap = make_snapshot(losses_dict, lbfgs_iter)
            save_checkpoint(
                model, lbfgs_checkpoint_path, history + [snap],
                {"phase": "lbfgs", "lbfgs_iter": lbfgs_iter, "loss": snap["loss"], "final": final},
                optimizer=lbfgs,
                extra_state={"iteration": snap["iter"], "lbfgs_iter": lbfgs_iter,
                             "best_loss": best_record["loss"],
                             "causal_eps": eps, "causal_bins": M, "lbfgs_mode": lbfgs_mode},
            )

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            losses_dict = build_causal_loss(model, data, binspec, eps, fixed_w=lbfgs_fixed_w)
            losses_dict["loss"].backward()
            step_counter["count"] += 1
            li = step_counter["count"]
            cur_true = losses_dict["loss_true"]
            if cur_true < lbfgs_best["loss"]:
                lbfgs_best["loss"] = cur_true
                lbfgs_best["iter"] = li
            if li == 1 or li % print_every == 0:
                el = time.time() - lbfgs_start
                rate = li / el if el > 0 else 0.0
                print(
                    "LBFGS %d/%d | total=%.3e true=%.3e f=%.3e ic=%.3e wall=%.3e inlet=%.3e "
                    "outlet=%.3e | min_w=%.3e | best=%.3e@%d | %.1f it/s"
                    % (li, lbfgs_steps, losses_dict["loss"].detach().cpu().item(), cur_true,
                       losses_dict["loss_f"].detach().cpu().item(),
                       losses_dict["loss_ic"].detach().cpu().item(),
                       losses_dict["loss_wall"].detach().cpu().item(),
                       losses_dict["loss_inlet"].detach().cpu().item(),
                       losses_dict["loss_outlet"].detach().cpu().item(),
                       losses_dict["min_w"], lbfgs_best["loss"], lbfgs_best["iter"], rate),
                    flush=True,
                )
                history.append(make_snapshot(losses_dict, li))  # persist for the loss curve
            if lbfgs_save_every > 0 and li % lbfgs_save_every == 0:
                save_lbfgs_ckpt(losses_dict, li)
            return losses_dict["loss"]

        # Single call: L-BFGS iterates internally and stops on its own tolerance
        # (gtd/change) or at max_iter — no manual restart, no state clearing.
        lbfgs.step(closure)

        final_losses = build_causal_loss(model, data, binspec, eps, fixed_w=lbfgs_fixed_w)
        final_snap = make_snapshot(final_losses, step_counter["count"])
        history.append(final_snap)
        if final_snap["loss_true"] < best_record["loss"]:
            best_record = {"loss": final_snap["loss_true"], "iter": final_snap["iter"]}
        save_lbfgs_ckpt(final_losses, step_counter["count"], final=True)
        print("[lbfgs] converged after %d closure evals (max_iter=%d)"
              % (step_counter["count"], lbfgs_steps), flush=True)

    last_logged = int(history[-1].get("iter", 0)) if history else 0
    print("=== Run summary | best true-loss=%.3e @ iter %d | last iter=%d | causality %s | "
          "wall=%s ===" % (best_record["loss"], best_record["iter"], last_logged,
          "satisfied" if causality_satisfied else "NOT reached",
          _fmt_hms(time.time() - start_time)), flush=True)
    print("--- %.2f seconds ---" % (time.time() - start_time), flush=True)
    final_opt_state = optimizer.state_dict()
    final_sched_state = scheduler.state_dict() if scheduler is not None else None
    return history, best_record, weight_history, final_opt_state, final_sched_state


def main():
    args = parse_args()

    ckpt_dir = f"results/checkpoints/{args.exp_name}"
    if not args.checkpoint:
        args.checkpoint = f"{ckpt_dir}/latest.pt"
    if not args.best_checkpoint:
        args.best_checkpoint = f"{ckpt_dir}/best.pt"
    if not args.lbfgs_checkpoint:
        args.lbfgs_checkpoint = f"{ckpt_dir}/latest_lbfgs.pt"
    if not args.output_dir:
        args.output_dir = f"results/figures/{args.exp_name}"
    if not args.loss_history:
        args.loss_history = f"results/logs/{args.exp_name}/loss_history.pkl"
    weight_history_path = f"results/logs/{args.exp_name}/causal_weights_history.pkl"
    weight_curve_path = f"results/figures/{args.exp_name}/causal_weight_curve.png"

    device = resolve_device(args.device)
    print("Using device:", device, flush=True)
    print(f"Experiment: {args.exp_name}  eps={args.causal_eps}  bins={args.causal_bins} "
          f"({args.causal_bin_mode})  lbfgs={args.causal_lbfgs}", flush=True)

    torch.manual_seed(1234)
    np.random.seed(1234)

    uv_layers = [3] + 7 * [50] + [5]
    data = build_training_data(device=device, tmax=args.tmax, period=args.period)
    model = PINNLaminarFlowTransient(
        uv_layers=uv_layers, lb=data["lb"], ub=data["ub"], mu=args.mu
    ).to(device)

    M = args.causal_bins
    bin_idx, safe_cnt, n_total = compute_bins(data["t_c"], M, args.tmax, args.causal_bin_mode, device)
    binspec = (bin_idx, safe_cnt, n_total, M)
    cnt_min, cnt_max = int(safe_cnt.min()), int(safe_cnt.max())
    print(f"[bins] mode={args.causal_bin_mode} M={M} N={n_total} "
          f"per-bin count: min={cnt_min} max={cnt_max}", flush=True)
    if args.causal_eps == 0.0:
        print("[causal] eps=0 → loss_f reduces exactly to vanilla mean residual.", flush=True)

    snapshot_iters = set()
    if args.snapshot_iters.strip():
        snapshot_iters = {int(x) for x in args.snapshot_iters.split(",") if x.strip()}
        if snapshot_iters:
            print(f"[snapshot] will save at Adam iters: {sorted(snapshot_iters)}", flush=True)

    optimizer_state = None
    scheduler_state = None
    start_iter = 1
    existing_history = []
    existing_weight_history = []
    best_loss_resume = None

    if args.load_checkpoint:
        ckpt = load_checkpoint(model, args.load_checkpoint, map_location=device)
        ckpt_mu = ckpt.get("config", {}).get("mu")
        if ckpt_mu is not None and ckpt_mu != args.mu:
            print(f"[mu] CLI --mu={args.mu} overridden by checkpoint mu={ckpt_mu}", flush=True)
            args.mu = ckpt_mu
            model.mu = ckpt_mu
        extra_state = ckpt.get("extra_state", {})
        raw_opt = ckpt.get("optimizer_state_dict")
        existing_history = list(ckpt.get("history", []))
        if raw_opt is not None and "lbfgs_iter" not in extra_state:
            optimizer_state = raw_opt
        if args.resume:
            scheduler_state = ckpt.get("scheduler_state_dict")
            if os.path.exists(weight_history_path):
                with open(weight_history_path, "rb") as f:
                    existing_weight_history = pickle.load(f)
            if "iteration" in extra_state:
                start_iter = int(extra_state.get("iteration", 0)) + 1
            elif existing_history:
                start_iter = int(existing_history[-1].get("iter", len(existing_history))) + 1
            best_loss_resume = extra_state.get("best_loss")
            print("Resuming from iteration %d" % start_iter, flush=True)

    freeze_weight_state = None
    if args.freeze_weight_checkpoint:
        fck = torch.load(args.freeze_weight_checkpoint, map_location=device, weights_only=False)
        freeze_weight_state = fck["model_state_dict"]
        print("[lbfgs] frozen weights will be computed from %s"
              % args.freeze_weight_checkpoint, flush=True)

    history, best_record, weight_history, final_opt_state, final_sched_state = train_causal(
        model=model, data=data, binspec=binspec, eps=args.causal_eps,
        adam_iters=args.adam_iters, learning_rate=args.learning_rate,
        lbfgs_steps=args.lbfgs_iters, print_every=args.print_every,
        lbfgs_mode=args.causal_lbfgs,
        scheduler_type=args.scheduler, scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma, scheduler_min_lr=args.scheduler_min_lr,
        scheduler_plateau_patience=args.scheduler_plateau_patience,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_warmup=args.early_stop_warmup,
        causality_delta=args.causality_delta,
        start_iter=start_iter, checkpoint_path=args.checkpoint,
        save_every=args.save_every, snapshot_iters=snapshot_iters,
        save_best=args.save_best, best_checkpoint_path=args.best_checkpoint,
        lbfgs_checkpoint_path=args.lbfgs_checkpoint, lbfgs_save_every=args.lbfgs_save_every,
        optimizer_state_dict=optimizer_state, scheduler_state_dict=scheduler_state,
        existing_history=existing_history,
        weight_history=existing_weight_history, initial_best_loss=best_loss_resume,
        freeze_weight_state=freeze_weight_state,
    )

    last_iter = int(history[-1].get("iter", len(history))) if history else (start_iter - 1)
    save_checkpoint(
        model, args.checkpoint, history,
        {
            "uv_layers": uv_layers, "lb": data["lb"].tolist(), "ub": data["ub"].tolist(),
            "adam_iters": args.adam_iters, "lbfgs_iters": args.lbfgs_iters,
            "learning_rate": args.learning_rate, "tmax": args.tmax, "mu": args.mu,
            "causal_eps": args.causal_eps, "causal_bins": M,
            "causal_bin_mode": args.causal_bin_mode, "causal_lbfgs": args.causal_lbfgs,
            "scheduler": args.scheduler,
            "best_iter": best_record["iter"], "best_loss": best_record["loss"],
        },
        extra_state={"iteration": last_iter, "best_loss": best_record["loss"],
                     "causal_eps": args.causal_eps, "causal_bins": M},
        optimizer_state_dict=final_opt_state,
        scheduler_state_dict=final_sched_state,
    )

    log_dir = os.path.dirname(os.path.abspath(args.loss_history))
    os.makedirs(log_dir, exist_ok=True)
    with open(args.loss_history, "wb") as f:
        pickle.dump(history, f)
    with open(weight_history_path, "wb") as f:
        pickle.dump(weight_history, f)

    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Causality diagnostics: min_w vs iter + final per-bin weight profile.
    if weight_history:
        iters = [e["iter"] for e in weight_history]
        min_w = [e["min_w"] for e in weight_history]
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 4))
        ax1.plot(iters, min_w)
        ax1.set_xlabel("Adam iteration")
        ax1.set_ylabel("min_i w_i")
        ax1.set_title(f"Causality indicator — {args.exp_name} (ε={args.causal_eps})")
        ax1.set_ylim([-0.05, 1.05])
        final_w = weight_history[-1]["weights"]
        ax2.plot(range(len(final_w)), final_w, marker="o", ms=3)
        ax2.set_xlabel("temporal bin (early → late)")
        ax2.set_ylabel("w_i")
        ax2.set_title("Final causal weight profile")
        ax2.set_ylim([-0.05, 1.05])
        fig.tight_layout()
        fig.savefig(weight_curve_path, dpi=150)
        plt.close(fig)
        print(f"Causal weight curve saved to {weight_curve_path}", flush=True)

    # Field frames (same as unsteady.py)
    t_front = np.linspace(0, args.tmax, 100).reshape(-1, 1)
    x_front = np.full_like(t_front, 0.15)
    y_front = np.full_like(t_front, 0.20)
    _, _, p_front = model.predict(x_front, y_front, t_front)
    plt.figure()
    plt.plot(t_front, p_front)
    plt.title("Pressure at leading point")
    plt.xlabel("t")
    plt.ylabel("p")
    plt.savefig(os.path.join(args.output_dir, "front_pressure.png"), dpi=150)
    plt.close()

    x_star = np.linspace(0, 1.1, 401)
    y_star = np.linspace(0, 0.41, 161)
    x_star, y_star = np.meshgrid(x_star, y_star)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]
    dst = np.sqrt((x_star - 0.2) ** 2 + (y_star - 0.2) ** 2)
    x_star = x_star[dst >= 0.05].reshape(-1, 1)
    y_star = y_star[dst >= 0.05].reshape(-1, 1)
    for i in range(args.num_frames):
        t_star = np.full((x_star.shape[0], 1), i * args.tmax / (args.num_frames - 1))
        u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
        post_process(
            xmin=0, xmax=1.1, ymin=0, ymax=0.41,
            field=[x_star, y_star, t_star, u_pred, v_pred, p_pred],
            out_path=os.path.join(args.output_dir, "field_frame_%03d.png" % i),
            s=2, title="Time: %.3fs" % (i * args.tmax / (args.num_frames - 1)),
        )


if __name__ == "__main__":
    main()
