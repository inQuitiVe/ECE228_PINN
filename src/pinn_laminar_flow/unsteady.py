import argparse
import os
import pickle
import shutil
import time

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from pyDOE import lhs


def gradients(y, x):
    return torch.autograd.grad(
        y,
        x,
        grad_outputs=torch.ones_like(y),
        create_graph=True,
        retain_graph=True,
    )[0]


class MLP(nn.Module):
    def __init__(self, layers):
        super().__init__()
        modules = []
        for in_features, out_features in zip(layers[:-2], layers[1:-1]):
            modules.append(nn.Linear(in_features, out_features))
            modules.append(nn.Tanh())
        modules.append(nn.Linear(layers[-2], layers[-1]))
        self.network = nn.Sequential(*modules)
        self._reset_parameters()

    def _reset_parameters(self):
        for module in self.network:
            if isinstance(module, nn.Linear):
                nn.init.xavier_normal_(module.weight)
                nn.init.zeros_(module.bias)

    def forward(self, x):
        return self.network(x)


class PINNLaminarFlowTransient(nn.Module):
    def __init__(self, uv_layers, lb, ub, rho=1.0, mu=0.005):
        super().__init__()
        self.rho = rho
        self.mu = mu
        self.lb = torch.as_tensor(lb, dtype=torch.float32).reshape(1, -1)
        self.ub = torch.as_tensor(ub, dtype=torch.float32).reshape(1, -1)
        self.network = MLP(uv_layers)

    def net_uv(self, x, y, t):
        inputs = torch.cat([x, y, t], dim=1)
        lb = self.lb.to(inputs.device)
        ub = self.ub.to(inputs.device)
        inputs = 2.0 * (inputs - lb) / (ub - lb) - 1.0
        outputs = self.network(inputs)
        psi = outputs[:, 0:1]
        p = outputs[:, 1:2]
        s11 = outputs[:, 2:3]
        s22 = outputs[:, 3:4]
        s12 = outputs[:, 4:5]
        u = gradients(psi, y)
        v = -gradients(psi, x)
        return u, v, p, s11, s22, s12

    def net_f(self, x, y, t):
        u, v, p, s11, s22, s12 = self.net_uv(x, y, t)

        s11_x = gradients(s11, x)
        s12_y = gradients(s12, y)
        s22_y = gradients(s22, y)
        s12_x = gradients(s12, x)

        u_x = gradients(u, x)
        u_y = gradients(u, y)
        v_x = gradients(v, x)
        v_y = gradients(v, y)
        u_t = gradients(u, t)
        v_t = gradients(v, t)

        f_u = self.rho * u_t + self.rho * (u * u_x + v * u_y) - s11_x - s12_y
        f_v = self.rho * v_t + self.rho * (u * v_x + v * v_y) - s12_x - s22_y
        f_s11 = -p + 2.0 * self.mu * u_x - s11
        f_s22 = -p + 2.0 * self.mu * v_y - s22
        f_s12 = self.mu * (u_y + v_x) - s12
        f_p = p + (s11 + s22) / 2.0
        return f_u, f_v, f_s11, f_s22, f_s12, f_p

    def predict(self, x, y, t):
        device = next(self.parameters()).device
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1, 1).requires_grad_(True)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(-1, 1).requires_grad_(True)
        t_t = torch.as_tensor(t, dtype=torch.float32, device=device).reshape(-1, 1).requires_grad_(True)
        self.eval()
        with torch.enable_grad():
            u, v, p, _, _, _ = self.net_uv(x_t, y_t, t_t)
        return u.detach().cpu().numpy(), v.detach().cpu().numpy(), p.detach().cpu().numpy()


def to_tensor(array, requires_grad=False, device="cpu"):
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
    if requires_grad:
        tensor = tensor.clone().detach().requires_grad_(True)
    return tensor


def mse_zero(x):
    return torch.mean(x.square())


def del_src_pt(xy_c, xc=0.0, yc=0.0, r=0.1):
    dst = np.sqrt((xy_c[:, 0] - xc) ** 2 + (xy_c[:, 1] - yc) ** 2)
    return xy_c[dst > r, :]


def cart_grid(xmin, xmax, ymin, ymax, tmin, tmax, num_x, num_y, num_t):
    x = np.linspace(xmin, xmax, num=num_x)
    y = np.linspace(ymin, ymax, num=num_y)
    t = np.linspace(tmin, tmax, num=num_t)
    xx, yy, tt = np.meshgrid(x, y, t)
    return xx.flatten()[:, None], yy.flatten()[:, None], tt.flatten()[:, None]


def gen_circle_pt(xc, yc, r, tmin, tmax, num_r, num_t):
    theta = np.linspace(0.0, np.pi * 2.0, num_r)
    x = r * np.cos(theta) + xc
    y = r * np.sin(theta) + yc
    t = np.linspace(tmin, tmax, num_t)
    xx, tt = np.meshgrid(x, t)
    yy, _ = np.meshgrid(y, t)
    return xx.flatten()[:, None], yy.flatten()[:, None], tt.flatten()[:, None]


def build_training_data(device, tmax=0.5):
    xmax = 1.1
    lb = np.array([0.0, 0.0, 0.0], dtype=np.float32)
    ub = np.array([xmax, 0.41, tmax], dtype=np.float32)

    x_ic, y_ic, t_ic = cart_grid(0, xmax, 0, 0.41, 0, 0, 81, 41, 1)
    ic = np.concatenate((x_ic, y_ic, t_ic), axis=1)
    ic = del_src_pt(ic, xc=0.2, yc=0.2, r=0.05)

    x_upb, y_upb, t_upb = cart_grid(0, xmax, 0.41, 0.41, 0, tmax, 81, 1, 41)
    x_lwb, y_lwb, t_lwb = cart_grid(0, xmax, 0, 0, 0, tmax, 81, 1, 41)
    wall_up = np.concatenate((x_upb, y_upb, t_upb), axis=1)
    wall_lw = np.concatenate((x_lwb, y_lwb, t_lwb), axis=1)

    u_max = 0.5
    period = tmax * 2.0
    x_inb, y_inb, t_inb = cart_grid(0, 0, 0, 0.41, 0, tmax, 1, 61, 61)
    u_inb = 4.0 * u_max * y_inb * (0.41 - y_inb) / (0.41 ** 2) * (np.sin(2.0 * np.pi * t_inb / period + 1.5 * np.pi) + 1.0)
    v_inb = np.zeros_like(x_inb)
    inb = np.concatenate((x_inb, y_inb, t_inb, u_inb, v_inb), axis=1)

    x_outb, y_outb, t_outb = cart_grid(1.1, 1.1, 0, 0.41, 0, tmax, 1, 81, 41)
    outb = np.concatenate((x_outb, y_outb, t_outb), axis=1)

    r = 0.05
    x_surf, y_surf, t_surf = gen_circle_pt(0.2, 0.2, r, 0, tmax, 81, 51)
    hole = np.concatenate((x_surf, y_surf, t_surf), axis=1)
    wall = np.concatenate((hole, wall_up, wall_lw), axis=0)

    xy_c = lb + (ub - lb) * lhs(3, 80000)
    xy_c_refine = np.array([0.0, 0.0, 0.0], dtype=np.float32) + np.array([0.4, 0.4, tmax], dtype=np.float32) * lhs(3, 15000)
    xy_c_lw = np.array([0.0, 0.0, 0.0], dtype=np.float32) + np.array([1.1, 0.02, tmax], dtype=np.float32) * lhs(3, 3000)
    xy_c_up = np.array([0.0, 0.39, 0.0], dtype=np.float32) + np.array([1.1, 0.02, tmax], dtype=np.float32) * lhs(3, 3000)
    xy_c = np.concatenate((xy_c, xy_c_refine, xy_c_lw, xy_c_up), axis=0)
    xy_c = del_src_pt(xy_c, xc=0.2, yc=0.2, r=0.05)
    xy_c = np.concatenate((xy_c, wall, outb, inb[:, 0:3]), axis=0)

    return {
        "lb": lb,
        "ub": ub,
        "x_c": to_tensor(xy_c[:, 0:1], requires_grad=True, device=device),
        "y_c": to_tensor(xy_c[:, 1:2], requires_grad=True, device=device),
        "t_c": to_tensor(xy_c[:, 2:3], requires_grad=True, device=device),
        "x_ic": to_tensor(ic[:, 0:1], requires_grad=True, device=device),
        "y_ic": to_tensor(ic[:, 1:2], requires_grad=True, device=device),
        "t_ic": to_tensor(ic[:, 2:3], requires_grad=True, device=device),
        "x_wall": to_tensor(wall[:, 0:1], requires_grad=True, device=device),
        "y_wall": to_tensor(wall[:, 1:2], requires_grad=True, device=device),
        "t_wall": to_tensor(wall[:, 2:3], requires_grad=True, device=device),
        "x_inlet": to_tensor(inb[:, 0:1], requires_grad=True, device=device),
        "y_inlet": to_tensor(inb[:, 1:2], requires_grad=True, device=device),
        "t_inlet": to_tensor(inb[:, 2:3], requires_grad=True, device=device),
        "u_inlet": to_tensor(inb[:, 3:4], device=device),
        "v_inlet": to_tensor(inb[:, 4:5], device=device),
        "x_outlet": to_tensor(outb[:, 0:1], requires_grad=True, device=device),
        "y_outlet": to_tensor(outb[:, 1:2], requires_grad=True, device=device),
        "t_outlet": to_tensor(outb[:, 2:3], requires_grad=True, device=device),
    }


def build_loss(model, data):
    f_u, f_v, f_s11, f_s22, f_s12, f_p = model.net_f(data["x_c"], data["y_c"], data["t_c"])
    u_ic, v_ic, p_ic, _, _, _ = model.net_uv(data["x_ic"], data["y_ic"], data["t_ic"])
    u_wall, v_wall, _, _, _, _ = model.net_uv(data["x_wall"], data["y_wall"], data["t_wall"])
    u_inlet, v_inlet, _, _, _, _ = model.net_uv(data["x_inlet"], data["y_inlet"], data["t_inlet"])
    _, _, p_outlet, _, _, _ = model.net_uv(data["x_outlet"], data["y_outlet"], data["t_outlet"])

    loss_f = mse_zero(f_u) + mse_zero(f_v) + mse_zero(f_s11) + mse_zero(f_s22) + mse_zero(f_s12) + mse_zero(f_p)
    loss_ic = mse_zero(u_ic) + mse_zero(v_ic) + mse_zero(p_ic)
    loss_wall = mse_zero(u_wall) + mse_zero(v_wall)
    loss_inlet = torch.mean((u_inlet - data["u_inlet"]).square()) + torch.mean((v_inlet - data["v_inlet"]).square())
    loss_outlet = mse_zero(p_outlet)
    loss = loss_f + 5.0 * loss_wall + 5.0 * loss_inlet + loss_outlet + loss_ic
    return {
        "loss": loss,
        "loss_f": loss_f,
        "loss_ic": loss_ic,
        "loss_wall": loss_wall,
        "loss_inlet": loss_inlet,
        "loss_outlet": loss_outlet,
    }


def train(
    model,
    data,
    adam_iters,
    learning_rate,
    lbfgs_steps,
    print_every,
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
    save_best=False,
    best_checkpoint_path=None,
    optimizer_state_dict=None,
    scheduler_state_dict=None,
    existing_history=None,
    initial_best_loss=None,
    initial_stale_steps=0,
):
    history = list(existing_history) if existing_history else []
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
            optimizer,
            mode="min",
            factor=scheduler_gamma,
            patience=max(1, scheduler_plateau_patience),
            min_lr=scheduler_min_lr,
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
        losses["loss"].backward()
        optimizer.step()
        if scheduler is not None:
            if scheduler_type == "plateau":
                scheduler.step(losses["loss"].detach().cpu().item())
            else:
                scheduler.step()

        snapshot = {name: value.detach().cpu().item() for name, value in losses.items()}
        snapshot["iter"] = iteration
        history.append(snapshot)
        if iteration == 1 or iteration % print_every == 0:
            print(
                "Adam %d | total=%.3e f=%.3e ic=%.3e wall=%.3e inlet=%.3e outlet=%.3e lr=%.3e"
                % (
                    iteration,
                    snapshot["loss"],
                    snapshot["loss_f"],
                    snapshot["loss_ic"],
                    snapshot["loss_wall"],
                    snapshot["loss_inlet"],
                    snapshot["loss_outlet"],
                    optimizer.param_groups[0]["lr"],
                )
            , flush=True)

        loss_value = snapshot["loss"]
        if loss_value < (best_loss - early_stop_min_delta):
            best_loss = loss_value
            stale_steps = 0
            if save_best and best_checkpoint_path:
                save_checkpoint(
                    model,
                    best_checkpoint_path,
                    history,
                    {"best_iter": iteration, "best_loss": loss_value},
                    optimizer=optimizer,
                    scheduler=scheduler,
                    extra_state={
                        "iteration": iteration,
                        "best_loss": loss_value,
                        "stale_steps": stale_steps,
                    },
                )
                best_record["loss"] = loss_value
                best_record["iter"] = iteration
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
                model,
                checkpoint_path,
                history,
                {"last_iter": iteration},
                optimizer=optimizer,
                scheduler=scheduler,
                extra_state={
                    "iteration": iteration,
                    "best_loss": best_loss,
                    "stale_steps": stale_steps,
                },
            )

    if lbfgs_steps > 0:
        lbfgs = torch.optim.LBFGS(
            model.parameters(),
            lr=1.0,
            max_iter=lbfgs_steps,
            max_eval=lbfgs_steps,
            history_size=50,
            line_search_fn="strong_wolfe",
        )
        step_counter = {"count": 0}

        def closure():
            lbfgs.zero_grad(set_to_none=True)
            losses = build_loss(model, data)
            losses["loss"].backward()
            step_counter["count"] += 1
            if step_counter["count"] == 1 or step_counter["count"] % print_every == 0:
                print(
                    "LBFGS %d | total=%.3e f=%.3e ic=%.3e wall=%.3e inlet=%.3e outlet=%.3e"
                    % (
                        step_counter["count"],
                        losses["loss"].detach().cpu().item(),
                        losses["loss_f"].detach().cpu().item(),
                        losses["loss_ic"].detach().cpu().item(),
                        losses["loss_wall"].detach().cpu().item(),
                        losses["loss_inlet"].detach().cpu().item(),
                        losses["loss_outlet"].detach().cpu().item(),
                    )
                , flush=True)
            return losses["loss"]

        lbfgs.step(closure)
        final_losses = build_loss(model, data)
        history.append({name: value.detach().cpu().item() for name, value in final_losses.items()})

    print("--- %.2f seconds ---" % (time.time() - start_time), flush=True)
    return history, best_record, stale_steps


def post_process(xmin, xmax, ymin, ymax, field, out_path, s=2, title=""):
    x_pred, y_pred, _, u_pred, v_pred, p_pred = field
    fig, ax = plt.subplots(nrows=3, figsize=(6, 8))

    cf = ax[0].scatter(x_pred, y_pred, c=u_pred, alpha=0.7, edgecolors="none", cmap="rainbow", marker="o", s=s, vmin=0, vmax=1.4)
    ax[0].axis("square")
    ax[0].set_xlim([xmin, xmax])
    ax[0].set_ylim([ymin, ymax])
    ax[0].set_title("u predict")
    fig.colorbar(cf, ax=ax[0], fraction=0.046, pad=0.04)

    cf = ax[1].scatter(x_pred, y_pred, c=v_pred, alpha=0.7, edgecolors="none", cmap="rainbow", marker="o", s=s, vmin=-0.7, vmax=0.7)
    ax[1].axis("square")
    ax[1].set_xlim([xmin, xmax])
    ax[1].set_ylim([ymin, ymax])
    ax[1].set_title("v predict")
    fig.colorbar(cf, ax=ax[1], fraction=0.046, pad=0.04)

    cf = ax[2].scatter(x_pred, y_pred, c=p_pred, alpha=0.7, edgecolors="none", cmap="rainbow", marker="o", s=s, vmin=-0.2, vmax=3)
    ax[2].axis("square")
    ax[2].set_xlim([xmin, xmax])
    ax[2].set_ylim([ymin, ymax])
    ax[2].set_title("p predict")
    fig.colorbar(cf, ax=ax[2], fraction=0.046, pad=0.04)

    if title:
        plt.suptitle(title, fontsize=16)
    plt.savefig(out_path, dpi=150)
    plt.close(fig)


def save_checkpoint(model, path, history, config, optimizer=None, scheduler=None, extra_state=None):
    checkpoint_dir = os.path.dirname(path)
    if checkpoint_dir:
        os.makedirs(checkpoint_dir, exist_ok=True)
    payload = {
        "model_state_dict": model.state_dict(),
        "history": history,
        "config": config,
    }
    if optimizer is not None:
        payload["optimizer_state_dict"] = optimizer.state_dict()
    if scheduler is not None:
        payload["scheduler_state_dict"] = scheduler.state_dict()
    if extra_state is not None:
        payload["extra_state"] = extra_state
    torch.save(
        payload,
        path,
    )
    print("Saved PyTorch checkpoint to %s" % path, flush=True)


def load_checkpoint(model, path, map_location):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded PyTorch checkpoint from %s" % path, flush=True)
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Transient mixed-form PINN in PyTorch")
    parser.add_argument("--adam-iters", type=int, default=5000)
    parser.add_argument("--lbfgs-iters", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--tmax", type=float, default=0.5)
    parser.add_argument("--checkpoint", default="results/unsteady/checkpoints/latest.pt")
    parser.add_argument("--best-checkpoint", default="results/unsteady/checkpoints/best.pt")
    parser.add_argument("--load-checkpoint", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--output-dir", default="results/unsteady/figures")
    parser.add_argument("--loss-history", default="results/unsteady/logs/loss_history.pkl")
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
    return parser.parse_args()


def resolve_device(device_arg):
    if device_arg == "auto":
        if torch.cuda.is_available():
            return torch.device("cuda")
        if torch.backends.mps.is_available():
            return torch.device("mps")
        return torch.device("cpu")
    return torch.device(device_arg)


def main():
    args = parse_args()
    device = resolve_device(args.device)
    print("Using device:", device, flush=True)

    torch.manual_seed(1234)
    np.random.seed(1234)

    uv_layers = [3] + 7 * [50] + [5]
    data = build_training_data(device=device, tmax=args.tmax)
    model = PINNLaminarFlowTransient(uv_layers=uv_layers, lb=data["lb"], ub=data["ub"]).to(device)

    optimizer_state = None
    scheduler_state = None
    start_iter = 1
    existing_history = []
    best_loss_resume = None
    stale_steps_resume = 0
    if args.load_checkpoint:
        ckpt = load_checkpoint(model, args.load_checkpoint, map_location=device)
        optimizer_state = ckpt.get("optimizer_state_dict")
        scheduler_state = ckpt.get("scheduler_state_dict")
        existing_history = list(ckpt.get("history", []))
        if args.resume:
            extra_state = ckpt.get("extra_state", {})
            if "iteration" in extra_state:
                start_iter = int(extra_state.get("iteration", 0)) + 1
            elif existing_history:
                hist_last = existing_history[-1]
                if "iter" in hist_last:
                    start_iter = int(hist_last["iter"]) + 1
                else:
                    start_iter = len(existing_history) + 1
            best_loss_resume = extra_state.get("best_loss")
            stale_steps_resume = int(extra_state.get("stale_steps", 0))
            print("Resuming from iteration %d" % start_iter, flush=True)

    history, best_record, stale_steps = train(
        model=model,
        data=data,
        adam_iters=args.adam_iters,
        learning_rate=args.learning_rate,
        lbfgs_steps=args.lbfgs_iters,
        print_every=args.print_every,
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
        save_best=args.save_best,
        best_checkpoint_path=args.best_checkpoint,
        optimizer_state_dict=optimizer_state,
        scheduler_state_dict=scheduler_state,
        existing_history=existing_history,
        initial_best_loss=best_loss_resume,
        initial_stale_steps=stale_steps_resume,
    )

    last_iter = int(history[-1]["iter"]) if history else (start_iter - 1)

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
            "scheduler": args.scheduler,
            "scheduler_step_size": args.scheduler_step_size,
            "scheduler_gamma": args.scheduler_gamma,
            "scheduler_min_lr": args.scheduler_min_lr,
            "scheduler_plateau_patience": args.scheduler_plateau_patience,
            "early_stop_patience": args.early_stop_patience,
            "early_stop_min_delta": args.early_stop_min_delta,
            "early_stop_warmup": args.early_stop_warmup,
            "best_iter": best_record["iter"],
            "best_loss": best_record["loss"],
        },
        extra_state={
            "iteration": last_iter,
            "best_loss": best_record["loss"],
            "stale_steps": stale_steps,
        },
    )

    loss_history_dir = os.path.dirname(os.path.abspath(args.loss_history))
    if loss_history_dir:
        os.makedirs(loss_history_dir, exist_ok=True)
    with open(args.loss_history, "wb") as handle:
        pickle.dump(history, handle)

    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)

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
        field = [x_star, y_star, t_star, u_pred, v_pred, p_pred]
        post_process(
            xmin=0,
            xmax=1.1,
            ymin=0,
            ymax=0.41,
            field=field,
            out_path=os.path.join(args.output_dir, "field_frame_%03d.png" % i),
            s=2,
            title="Time: %.3fs" % (i * args.tmax / (args.num_frames - 1)),
        )


if __name__ == "__main__":
    main()
