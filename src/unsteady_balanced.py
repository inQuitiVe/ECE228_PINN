"""Transient PINN training with pluggable loss balancers.

Drop-in replacement for unsteady.py that adds --balancer {none, grad_norm}.
All model / data / checkpoint logic is shared with unsteady.py via direct import.
"""

import argparse
import os
import pickle
import shutil
import sys
import time

import matplotlib.pyplot as plt
import numpy as np
import torch

# Shared PINN utilities from unsteady.py
from unsteady import (
    PINNLaminarFlowTransient,
    build_loss,
    build_training_data,
    load_checkpoint,
    post_process,
    resolve_device,
    save_checkpoint,
)
from loss_balancers import make_balancer


def parse_args():
    parser = argparse.ArgumentParser(
        description="Transient mixed-form PINN with loss balancer"
    )
    # ---- identical to unsteady.py parse_args ----
    parser.add_argument("--adam-iters", type=int, default=10000)
    parser.add_argument("--lbfgs-iters", type=int, default=50000)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--tmax", type=float, default=0.5)
    parser.add_argument("--mu", type=float, default=0.005,
                        help="Dynamic viscosity (Re=10: 0.005, Re=100: 0.0005)")
    parser.add_argument("--period", type=float, default=1.0)
    parser.add_argument("--exp-name", default="grad_norm",
                        help="Experiment name; paths scoped under results/{...}/{exp_name}/")
    parser.add_argument("--checkpoint", default="")
    parser.add_argument("--best-checkpoint", default="")
    parser.add_argument("--lbfgs-checkpoint", default="")
    parser.add_argument("--load-checkpoint", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--lbfgs-save-every", type=int, default=500)
    parser.add_argument("--output-dir", default="")
    parser.add_argument("--loss-history", default="")
    parser.add_argument("--num-frames", type=int, default=51)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--scheduler", default="none", choices=["none", "cosine", "step", "plateau"])
    parser.add_argument("--scheduler-step-size", type=int, default=1000)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-5)
    parser.add_argument("--scheduler-plateau-patience", type=int, default=200)
    parser.add_argument("--early-stop-patience", type=int, default=0)
    parser.add_argument("--early-stop-min-delta", type=float, default=0.0)
    parser.add_argument("--early-stop-warmup", type=int, default=0)
    # ---- balancer-specific ----
    parser.add_argument("--balancer", default="grad_norm",
                        choices=["none", "grad_norm", "strict_grad_norm"],
                        help="Loss balancer: none (fixed β), grad_norm (L2-norm ratio variant), "
                             "strict_grad_norm (paper-faithful Wang 2020 Alg.1, max|∇L_r|/mean|∇L_i|)")
    parser.add_argument("--balancer-alpha", type=float, default=None,
                        help="EMA decay (grad_norm default 0.1, strict_grad_norm default 0.9 per paper)")
    parser.add_argument("--balancer-update-every", type=int, default=10,
                        help="Update λ every N Adam iterations (N >= 1)")
    parser.add_argument("--balancer-clip-min", type=float, default=1e-2,
                        help="Lower clip for λ̂ (strict_grad_norm only)")
    parser.add_argument("--balancer-clip-max", type=float, default=1e3,
                        help="Upper clip for λ̂ (strict_grad_norm only)")
    parser.add_argument("--snapshot-iters", type=str, default="",
                        help="Comma-separated Adam iters at which to save NON-overwriting "
                             "snapshot_iter{N}.pt (e.g., '10000,20000')")
    args = parser.parse_args()
    if args.balancer_update_every < 1:
        parser.error("--balancer-update-every must be >= 1")
    return args


def train_balanced(
    model,
    data,
    balancer,
    adam_iters,
    learning_rate,
    lbfgs_steps,
    print_every,
    balancer_update_every=10,
    early_stop_patience=0,
    early_stop_min_delta=0.0,
    early_stop_warmup=0,
    scheduler_type="none",
    scheduler_step_size=1000,
    scheduler_gamma=0.5,
    scheduler_min_lr=1e-5,
    scheduler_plateau_patience=200,
    start_iter=1,
    checkpoint_path=None,
    save_every=0,
    snapshot_iters=None,
    save_best=False,
    best_checkpoint_path=None,
    lbfgs_checkpoint_path=None,
    lbfgs_save_every=0,
    optimizer_state_dict=None,
    scheduler_state_dict=None,
    lbfgs_optimizer_state_dict=None,
    existing_history=None,
    initial_best_loss=None,
    initial_stale_steps=0,
    lambda_history=None,
):
    history = list(existing_history) if existing_history else []
    lambda_history = list(lambda_history) if lambda_history else []

    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    if scheduler_type == "cosine":
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=max(1, adam_iters), eta_min=scheduler_min_lr
        )
    elif scheduler_type == "step":
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer, step_size=max(1, scheduler_step_size), gamma=scheduler_gamma
        )
    elif scheduler_type == "plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, mode="min", factor=scheduler_gamma,
            patience=max(1, scheduler_plateau_patience), min_lr=scheduler_min_lr,
        )
    else:
        scheduler = None
    if optimizer_state_dict is not None:
        optimizer.load_state_dict(optimizer_state_dict)
    if scheduler is not None and scheduler_state_dict is not None:
        scheduler.load_state_dict(scheduler_state_dict)

    start_time = time.time()
    model.train()
    best_loss = float(initial_best_loss) if initial_best_loss is not None else float("inf")
    best_record = {"loss": best_loss, "iter": 0}
    if history:
        for idx, item in enumerate(history, start=1):
            if "iter" not in item:
                item["iter"] = idx
        best_entry = min(history, key=lambda item: item["loss"])
        if best_entry["loss"] < best_record["loss"]:
            best_record["loss"] = best_entry["loss"]
            best_record["iter"] = int(best_entry["iter"])
    stale_steps = int(initial_stale_steps)

    for iteration in range(start_iter, adam_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = build_loss(model, data)

        # Update λ weights (measures grad norms per term, zeros grads when done)
        if iteration % balancer_update_every == 0:
            balancer.update(losses, list(model.parameters()))
            lam_entry = {"iter": iteration, **balancer.log_lambdas()}
            lambda_history.append(lam_entry)

        total = balancer.compute_total(losses)
        total.backward()
        optimizer.step()

        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(total.detach().cpu().item())
            else:
                scheduler.step()

        snapshot = {name: value.detach().cpu().item() for name, value in losses.items()}
        snapshot["loss"] = total.detach().cpu().item()
        snapshot["iter"] = iteration
        history.append(snapshot)

        if iteration == 1 or iteration % print_every == 0:
            lambdas = balancer.log_lambdas()
            lam_str = " ".join(f"{k.replace('loss_','')}={v:.2f}" for k, v in lambdas.items())
            print(
                "Adam %d | total=%.3e f=%.3e ic=%.3e wall=%.3e inlet=%.3e outlet=%.3e lr=%.3e | λ[%s]"
                % (
                    iteration,
                    snapshot["loss"],
                    losses["loss_f"].detach().cpu().item(),
                    losses["loss_ic"].detach().cpu().item(),
                    losses["loss_wall"].detach().cpu().item(),
                    losses["loss_inlet"].detach().cpu().item(),
                    losses["loss_outlet"].detach().cpu().item(),
                    optimizer.param_groups[0]["lr"],
                    lam_str,
                ),
                flush=True,
            )

        loss_value = snapshot["loss"]
        if loss_value < (best_loss - early_stop_min_delta):
            best_loss = loss_value
            stale_steps = 0
            best_record = {"loss": loss_value, "iter": iteration}
            if save_best and best_checkpoint_path:
                save_checkpoint(
                    model, best_checkpoint_path, history,
                    {"best_iter": iteration, "best_loss": loss_value},
                    optimizer=optimizer, scheduler=scheduler,
                    extra_state={
                        "iteration": iteration, "best_loss": loss_value,
                        "stale_steps": stale_steps,
                        "balancer_state": balancer.state_dict(),
                    },
                )
        elif iteration > early_stop_warmup:
            stale_steps += 1
            if early_stop_patience > 0 and stale_steps >= early_stop_patience:
                print(
                    "Early stop at Adam iteration %d (best=%.3e, current=%.3e)"
                    % (iteration, best_loss, loss_value),
                    flush=True,
                )
                break

        if save_every > 0 and checkpoint_path and iteration % save_every == 0:
            save_checkpoint(
                model, checkpoint_path, history, {"last_iter": iteration},
                optimizer=optimizer, scheduler=scheduler,
                extra_state={
                    "iteration": iteration, "best_loss": best_loss,
                    "stale_steps": stale_steps,
                    "balancer_state": balancer.state_dict(),
                },
            )

        if snapshot_iters and iteration in snapshot_iters and checkpoint_path:
            snap_path = os.path.join(os.path.dirname(checkpoint_path),
                                     f"snapshot_iter{iteration}.pt")
            save_checkpoint(
                model, snap_path, history,
                {"snapshot_iter": iteration, "phase": "adam"},
                optimizer=optimizer, scheduler=scheduler,
                extra_state={
                    "iteration": iteration, "best_loss": best_loss,
                    "stale_steps": stale_steps,
                    "balancer_state": balancer.state_dict(),
                },
            )
            print(f"[snapshot] iter {iteration} → {snap_path}", flush=True)

    # ---- L-BFGS phase (λ frozen) ----
    if lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(), lr=1.0, max_iter=lbfgs_steps, max_eval=lbfgs_steps,
            history_size=50, tolerance_change=0, tolerance_grad=0,
            line_search_fn="strong_wolfe",
        )
        if lbfgs_optimizer_state_dict is not None:
            lbfgs.load_state_dict(lbfgs_optimizer_state_dict)
            for pg_state in lbfgs.state.values():
                pg_state["n_iter"] = 0
                pg_state["prev_flat_grad"] = None
                pg_state.pop("prev_loss", None)

        step_counter = {"count": 0}
        last_iter = max((int(item.get("iter", 0)) for item in history), default=start_iter - 1)

        def make_lbfgs_snapshot(losses_dict, total_val, lbfgs_iter):
            snapshot = {name: value.detach().cpu().item() for name, value in losses_dict.items()}
            snapshot["loss"] = total_val
            snapshot["iter"] = last_iter + lbfgs_iter
            snapshot["phase"] = "lbfgs"
            snapshot["lbfgs_iter"] = lbfgs_iter
            return snapshot

        def save_lbfgs_checkpoint(losses_dict, total_val, lbfgs_iter, final=False):
            if not lbfgs_checkpoint_path:
                return
            snapshot = make_lbfgs_snapshot(losses_dict, total_val, lbfgs_iter)
            save_checkpoint(
                model, lbfgs_checkpoint_path, history + [snapshot],
                {"phase": "lbfgs", "lbfgs_iter": lbfgs_iter, "loss": total_val, "final": final},
                optimizer=lbfgs,
                extra_state={
                    "iteration": snapshot["iter"], "lbfgs_iter": lbfgs_iter,
                    "current_loss": total_val, "best_loss": best_record["loss"],
                    "stale_steps": stale_steps,
                    "balancer_state": balancer.state_dict(),
                },
            )

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            losses_dict = build_loss(model, data)
            total = balancer.compute_total(losses_dict)
            total.backward()
            step_counter["count"] += 1
            lbfgs_iter = step_counter["count"]
            total_val = total.detach().cpu().item()
            if lbfgs_iter == 1 or lbfgs_iter % print_every == 0:
                print(
                    "LBFGS %d | total=%.3e f=%.3e ic=%.3e wall=%.3e inlet=%.3e outlet=%.3e"
                    % (
                        lbfgs_iter, total_val,
                        losses_dict["loss_f"].detach().cpu().item(),
                        losses_dict["loss_ic"].detach().cpu().item(),
                        losses_dict["loss_wall"].detach().cpu().item(),
                        losses_dict["loss_inlet"].detach().cpu().item(),
                        losses_dict["loss_outlet"].detach().cpu().item(),
                    ),
                    flush=True,
                )
            if lbfgs_save_every > 0 and lbfgs_iter % lbfgs_save_every == 0:
                save_lbfgs_checkpoint(losses_dict, total_val, lbfgs_iter)
            return total

        while step_counter["count"] < lbfgs_steps:
            remaining = lbfgs_steps - step_counter["count"]
            lbfgs.param_groups[0]["max_iter"] = remaining
            lbfgs.param_groups[0]["max_eval"] = remaining
            lbfgs.step(closure)
            if step_counter["count"] < lbfgs_steps:
                for pg_state in lbfgs.state.values():
                    pg_state.clear()

        final_losses = build_loss(model, data)
        final_total = balancer.compute_total(final_losses).detach().cpu().item()
        final_snapshot = make_lbfgs_snapshot(final_losses, final_total, step_counter["count"])
        history.append(final_snapshot)
        if final_snapshot["loss"] < best_record["loss"]:
            best_record = {"loss": final_snapshot["loss"], "iter": final_snapshot["iter"]}
        save_lbfgs_checkpoint(final_losses, final_total, step_counter["count"], final=True)

    print("--- %.2f seconds ---" % (time.time() - start_time), flush=True)
    final_optimizer_state = optimizer.state_dict()
    final_scheduler_state = scheduler.state_dict() if scheduler is not None else None
    return history, best_record, stale_steps, lambda_history, final_optimizer_state, final_scheduler_state


def main():
    args = parse_args()

    run_dir = f"results/phase4a_loss_balancing/{args.exp_name}"
    ckpt_dir = f"{run_dir}/checkpoints"
    if not args.checkpoint:
        args.checkpoint = f"{ckpt_dir}/latest.pt"
    if not args.best_checkpoint:
        args.best_checkpoint = f"{ckpt_dir}/best.pt"
    if not args.lbfgs_checkpoint:
        args.lbfgs_checkpoint = f"{ckpt_dir}/latest_lbfgs.pt"
    if not args.output_dir:
        args.output_dir = f"{run_dir}/figures"
    if not args.loss_history:
        args.loss_history = f"{run_dir}/logs/loss_history.pkl"
    lambda_history_path = f"{run_dir}/logs/lambda_history.pkl"
    lambda_curve_path = f"{run_dir}/figures/lambda_curve.png"

    device = resolve_device(args.device)
    print("Using device:", device, flush=True)
    print(f"Experiment: {args.exp_name}  balancer: {args.balancer}", flush=True)

    torch.manual_seed(1234)
    np.random.seed(1234)

    uv_layers = [3] + 7 * [50] + [5]
    data = build_training_data(device=device, tmax=args.tmax, period=args.period)
    model = PINNLaminarFlowTransient(
        uv_layers=uv_layers, lb=data["lb"], ub=data["ub"], mu=args.mu
    ).to(device)

    balancer_kwargs = {}
    if args.balancer == "grad_norm":
        balancer_kwargs["alpha"] = args.balancer_alpha if args.balancer_alpha is not None else 0.1
    elif args.balancer == "strict_grad_norm":
        balancer_kwargs["alpha"] = args.balancer_alpha if args.balancer_alpha is not None else 0.9
        balancer_kwargs["clip_min"] = args.balancer_clip_min
        balancer_kwargs["clip_max"] = args.balancer_clip_max
    balancer = make_balancer(args.balancer, **balancer_kwargs)

    snapshot_iters = set()
    if args.snapshot_iters.strip():
        snapshot_iters = {int(x) for x in args.snapshot_iters.split(",") if x.strip()}
        if snapshot_iters:
            print(f"[snapshot] will save non-overwriting snapshots at Adam iters: "
                  f"{sorted(snapshot_iters)}", flush=True)

    optimizer_state = None
    scheduler_state = None
    lbfgs_optimizer_state = None
    start_iter = 1
    existing_history = []
    existing_lambda_history = []
    best_loss_resume = None
    stale_steps_resume = 0

    if args.load_checkpoint:
        ckpt = load_checkpoint(model, args.load_checkpoint, map_location=device)
        ckpt_mu = ckpt.get("config", {}).get("mu")
        if ckpt_mu is not None and ckpt_mu != args.mu:
            print(f"[mu] CLI --mu={args.mu} overridden by checkpoint mu={ckpt_mu}", flush=True)
            args.mu = ckpt_mu
            model.mu = ckpt_mu
        raw_optimizer_state = ckpt.get("optimizer_state_dict")
        scheduler_state = ckpt.get("scheduler_state_dict")
        existing_history = list(ckpt.get("history", []))
        extra_state = ckpt.get("extra_state", {})
        if raw_optimizer_state is not None and "lbfgs_iter" not in extra_state:
            optimizer_state = raw_optimizer_state
        if "balancer_state" in extra_state:
            balancer.load_state_dict(extra_state["balancer_state"])
            print("Restored balancer state from checkpoint.", flush=True)
        if args.resume:
            if os.path.exists(lambda_history_path):
                with open(lambda_history_path, "rb") as f:
                    existing_lambda_history = pickle.load(f)
            if extra_state.get("lbfgs_iter") is not None:
                print("[resume] WARNING: checkpoint is from L-BFGS phase; L-BFGS state is NOT "
                      "restored. Training will skip Adam and run a fresh L-BFGS phase from these "
                      "weights.", flush=True)
            if "iteration" in extra_state:
                start_iter = int(extra_state.get("iteration", 0)) + 1
            elif existing_history:
                hist_last = existing_history[-1]
                start_iter = int(hist_last.get("iter", len(existing_history))) + 1
            best_loss_resume = extra_state.get("best_loss")
            stale_steps_resume = int(extra_state.get("stale_steps", 0))
            print("Resuming from iteration %d" % start_iter, flush=True)

    history, best_record, stale_steps, lambda_history, final_opt_state, final_sched_state = train_balanced(
        model=model,
        data=data,
        balancer=balancer,
        adam_iters=args.adam_iters,
        learning_rate=args.learning_rate,
        lbfgs_steps=args.lbfgs_iters,
        print_every=args.print_every,
        balancer_update_every=args.balancer_update_every,
        early_stop_patience=args.early_stop_patience,
        early_stop_min_delta=args.early_stop_min_delta,
        early_stop_warmup=args.early_stop_warmup,
        scheduler_type=args.scheduler,
        scheduler_step_size=args.scheduler_step_size,
        scheduler_gamma=args.scheduler_gamma,
        scheduler_min_lr=args.scheduler_min_lr,
        scheduler_plateau_patience=args.scheduler_plateau_patience,
        start_iter=start_iter,
        checkpoint_path=args.checkpoint,
        save_every=args.save_every,
        snapshot_iters=snapshot_iters,
        save_best=args.save_best,
        best_checkpoint_path=args.best_checkpoint,
        lbfgs_checkpoint_path=args.lbfgs_checkpoint,
        lbfgs_save_every=args.lbfgs_save_every,
        optimizer_state_dict=optimizer_state,
        scheduler_state_dict=scheduler_state,
        lbfgs_optimizer_state_dict=lbfgs_optimizer_state,
        existing_history=existing_history,
        initial_best_loss=best_loss_resume,
        initial_stale_steps=stale_steps_resume,
        lambda_history=existing_lambda_history,
    )

    last_iter = int(history[-1].get("iter", len(history))) if history else (start_iter - 1)

    save_checkpoint(
        model,
        args.checkpoint,
        history,
        {
            "uv_layers": uv_layers,
            "lb": data["lb"].tolist(),
            "ub": data["ub"].tolist(),
            "adam_iters": args.adam_iters,
            "lbfgs_iters": args.lbfgs_iters,
            "learning_rate": args.learning_rate,
            "tmax": args.tmax,
            "mu": args.mu,
            "balancer": args.balancer,
            "balancer_alpha": args.balancer_alpha,
            "best_iter": best_record["iter"],
            "best_loss": best_record["loss"],
        },
        extra_state={
            "iteration": last_iter,
            "best_loss": best_record["loss"],
            "stale_steps": stale_steps,
            "balancer_state": balancer.state_dict(),
        },
        optimizer_state_dict=final_opt_state,
        scheduler_state_dict=final_sched_state,
    )

    # Save loss history
    log_dir = os.path.dirname(os.path.abspath(args.loss_history))
    os.makedirs(log_dir, exist_ok=True)
    with open(args.loss_history, "wb") as f:
        pickle.dump(history, f)

    # Save lambda history
    with open(lambda_history_path, "wb") as f:
        pickle.dump(lambda_history, f)

    # Saturation summary (strict_grad_norm only)
    if hasattr(balancer, "saturation_summary"):
        sat = balancer.saturation_summary()
        if sat:
            print("[saturation] λ̂ clip-hit fractions over Adam phase:", flush=True)
            for k, v in sat.items():
                print(f"  {k:12s}  min_hit={v['min_frac']*100:5.1f}%  "
                      f"max_hit={v['max_frac']*100:5.1f}%", flush=True)

    # Wipe & recreate output dir BEFORE saving figures (else lambda_curve gets nuked)
    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)

    # Plot lambda curve
    if lambda_history:
        iters = [e["iter"] for e in lambda_history]
        keys = [k for k in lambda_history[0] if k != "iter"]
        fig, ax = plt.subplots(figsize=(8, 4))
        for k in keys:
            vals = [e[k] for e in lambda_history]
            ax.plot(iters, vals, label=k)
        ax.set_xlabel("Adam iteration")
        ax.set_ylabel("λ")
        ax.set_title(f"Loss weights — {args.exp_name}")
        ax.legend()
        ax.set_yscale("log")
        fig.tight_layout()
        fig.savefig(lambda_curve_path, dpi=150)
        plt.close(fig)
        print(f"Lambda curve saved to {lambda_curve_path}", flush=True)

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
            s=2,
            title="Time: %.3fs" % (i * args.tmax / (args.num_frames - 1)),
        )


if __name__ == "__main__":
    main()
