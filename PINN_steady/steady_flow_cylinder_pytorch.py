import argparse
import os
import pickle
import time

import matplotlib.pyplot as plt
import numpy as np
import scipy.io
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


class PINNLaminarFlow(nn.Module):
    def __init__(self, uv_layers, lb, ub, rho=1.0, mu=0.02):
        super().__init__()
        self.rho = rho
        self.mu = mu
        self.lb = torch.as_tensor(lb, dtype=torch.float32).reshape(1, -1)
        self.ub = torch.as_tensor(ub, dtype=torch.float32).reshape(1, -1)
        self.network = MLP(uv_layers)

    def net_uv(self, x, y):
        inputs = torch.cat([x, y], dim=1)
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

    def net_f(self, x, y):
        u, v, p, s11, s22, s12 = self.net_uv(x, y)

        s11_x = gradients(s11, x)
        s12_y = gradients(s12, y)
        s22_y = gradients(s22, y)
        s12_x = gradients(s12, x)

        u_x = gradients(u, x)
        u_y = gradients(u, y)
        v_x = gradients(v, x)
        v_y = gradients(v, y)

        f_u = self.rho * (u * u_x + v * u_y) - s11_x - s12_y
        f_v = self.rho * (u * v_x + v * v_y) - s12_x - s22_y
        f_s11 = -p + 2.0 * self.mu * u_x - s11
        f_s22 = -p + 2.0 * self.mu * v_y - s22
        f_s12 = self.mu * (u_y + v_x) - s12
        f_p = p + (s11 + s22) / 2.0
        return f_u, f_v, f_s11, f_s22, f_s12, f_p

    def predict(self, x, y):
        device = next(self.parameters()).device
        x_t = torch.as_tensor(x, dtype=torch.float32, device=device).reshape(-1, 1).requires_grad_(True)
        y_t = torch.as_tensor(y, dtype=torch.float32, device=device).reshape(-1, 1).requires_grad_(True)
        self.eval()
        with torch.enable_grad():
            u, v, p, _, _, _ = self.net_uv(x_t, y_t)
        return u.detach().cpu().numpy(), v.detach().cpu().numpy(), p.detach().cpu().numpy()


def mse_zero(x):
    return torch.mean(x.square())


def build_loss(model, data):
    x_c, y_c = data["x_c"], data["y_c"]
    x_wall, y_wall = data["x_wall"], data["y_wall"]
    x_inlet, y_inlet = data["x_inlet"], data["y_inlet"]
    u_inlet, v_inlet = data["u_inlet"], data["v_inlet"]
    x_outlet, y_outlet = data["x_outlet"], data["y_outlet"]

    f_u, f_v, f_s11, f_s22, f_s12, f_p = model.net_f(x_c, y_c)
    u_wall, v_wall, _, _, _, _ = model.net_uv(x_wall, y_wall)
    u_inlet_pred, v_inlet_pred, _, _, _, _ = model.net_uv(x_inlet, y_inlet)
    _, _, p_outlet_pred, _, _, _ = model.net_uv(x_outlet, y_outlet)

    loss_f = (
        mse_zero(f_u)
        + mse_zero(f_v)
        + mse_zero(f_s11)
        + mse_zero(f_s22)
        + mse_zero(f_s12)
        + mse_zero(f_p)
    )
    loss_wall = mse_zero(u_wall) + mse_zero(v_wall)
    loss_inlet = torch.mean((u_inlet_pred - u_inlet).square()) + torch.mean((v_inlet_pred - v_inlet).square())
    loss_outlet = mse_zero(p_outlet_pred)
    loss = loss_f + 2.0 * (loss_wall + loss_inlet + loss_outlet)
    return {
        "loss": loss,
        "loss_f": loss_f,
        "loss_wall": loss_wall,
        "loss_inlet": loss_inlet,
        "loss_outlet": loss_outlet,
    }


def to_tensor(array, requires_grad=False, device="cpu"):
    tensor = torch.as_tensor(array, dtype=torch.float32, device=device)
    if requires_grad:
        tensor = tensor.clone().detach().requires_grad_(True)
    return tensor


def del_cyl_pt(xy_c, xc=0.0, yc=0.0, r=0.1):
    dst = np.sqrt((xy_c[:, 0] - xc) ** 2 + (xy_c[:, 1] - yc) ** 2)
    return xy_c[dst > r, :]


def preprocess_reference(path):
    data = scipy.io.loadmat(path)
    x = data["x"].flatten()[:, None]
    y = data["y"].flatten()[:, None]
    p = data["p"].flatten()[:, None]
    vx = data["vx"].flatten()[:, None]
    vy = data["vy"].flatten()[:, None]
    return x, y, vx, vy, p


def post_process(xmin, xmax, ymin, ymax, field_fluent, field_pinn, output_path, s=2, alpha=0.5, marker="o"):
    x_fluent, y_fluent, u_fluent, v_fluent, p_fluent = field_fluent
    x_pinn, y_pinn, u_pinn, v_pinn, p_pinn = field_pinn

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7, 4))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    cf = ax[0, 0].scatter(x_pinn, y_pinn, c=u_pinn, alpha=alpha - 0.1, edgecolors="none", cmap="rainbow", marker=marker, s=int(s))
    ax[0, 0].axis("square")
    ax[0, 0].set_xticks([])
    ax[0, 0].set_yticks([])
    ax[0, 0].set_xlim([xmin, xmax])
    ax[0, 0].set_ylim([ymin, ymax])
    ax[0, 0].set_title(r"$u$ (m/s)")
    fig.colorbar(cf, ax=ax[0, 0], fraction=0.046, pad=0.04)

    cf = ax[1, 0].scatter(x_pinn, y_pinn, c=v_pinn, alpha=alpha - 0.1, edgecolors="none", cmap="rainbow", marker=marker, s=int(s))
    ax[1, 0].axis("square")
    ax[1, 0].set_xticks([])
    ax[1, 0].set_yticks([])
    ax[1, 0].set_xlim([xmin, xmax])
    ax[1, 0].set_ylim([ymin, ymax])
    ax[1, 0].set_title(r"$v$ (m/s)")
    fig.colorbar(cf, ax=ax[1, 0], fraction=0.046, pad=0.04)

    cf = ax[2, 0].scatter(x_pinn, y_pinn, c=p_pinn, alpha=alpha, edgecolors="none", cmap="rainbow", marker=marker, s=int(s), vmin=-0.25, vmax=4.0)
    ax[2, 0].axis("square")
    ax[2, 0].set_xticks([])
    ax[2, 0].set_yticks([])
    ax[2, 0].set_xlim([xmin, xmax])
    ax[2, 0].set_ylim([ymin, ymax])
    ax[2, 0].set_title("Pressure (Pa)")
    fig.colorbar(cf, ax=ax[2, 0], fraction=0.046, pad=0.04)

    cf = ax[0, 1].scatter(x_fluent, y_fluent, c=u_fluent, alpha=alpha, edgecolors="none", cmap="rainbow", marker=marker, s=s)
    ax[0, 1].axis("square")
    ax[0, 1].set_xticks([])
    ax[0, 1].set_yticks([])
    ax[0, 1].set_xlim([xmin, xmax])
    ax[0, 1].set_ylim([ymin, ymax])
    ax[0, 1].set_title(r"$u$ (m/s)")
    fig.colorbar(cf, ax=ax[0, 1], fraction=0.046, pad=0.04)

    cf = ax[1, 1].scatter(x_fluent, y_fluent, c=v_fluent, alpha=alpha, edgecolors="none", cmap="rainbow", marker=marker, s=s)
    ax[1, 1].axis("square")
    ax[1, 1].set_xticks([])
    ax[1, 1].set_yticks([])
    ax[1, 1].set_xlim([xmin, xmax])
    ax[1, 1].set_ylim([ymin, ymax])
    ax[1, 1].set_title(r"$v$ (m/s)")
    fig.colorbar(cf, ax=ax[1, 1], fraction=0.046, pad=0.04)

    cf = ax[2, 1].scatter(x_fluent, y_fluent, c=p_fluent, alpha=alpha, edgecolors="none", cmap="rainbow", marker=marker, s=s, vmin=-0.25, vmax=4.0)
    ax[2, 1].axis("square")
    ax[2, 1].set_xticks([])
    ax[2, 1].set_yticks([])
    ax[2, 1].set_xlim([xmin, xmax])
    ax[2, 1].set_ylim([ymin, ymax])
    ax[2, 1].set_title("Pressure (Pa)")
    fig.colorbar(cf, ax=ax[2, 1], fraction=0.046, pad=0.04)

    for row in ax:
        for subplot in row:
            for spine in subplot.spines.values():
                spine.set_visible(False)

    plt.savefig(output_path, dpi=300)
    plt.close(fig)


def build_training_data(device, n_collo=40000, n_refine=10000):
    lb = np.array([0.0, 0.0], dtype=np.float32)
    ub = np.array([1.1, 0.41], dtype=np.float32)

    wall_up = np.array([0.0, 0.41], dtype=np.float32) + np.array([1.1, 0.0], dtype=np.float32) * lhs(2, 441)
    wall_lw = np.array([0.0, 0.0], dtype=np.float32) + np.array([1.1, 0.0], dtype=np.float32) * lhs(2, 441)

    u_max = 1.0
    inlet = np.array([0.0, 0.0], dtype=np.float32) + np.array([0.0, 0.41], dtype=np.float32) * lhs(2, 201)
    y_inlet = inlet[:, 1:2]
    u_inlet = 4.0 * u_max * y_inlet * (0.41 - y_inlet) / (0.41 ** 2)
    v_inlet = np.zeros_like(y_inlet)
    inlet = np.concatenate((inlet, u_inlet, v_inlet), axis=1)

    outlet = np.array([1.1, 0.0], dtype=np.float32) + np.array([0.0, 0.41], dtype=np.float32) * lhs(2, 201)

    r = 0.05
    theta = np.array([0.0], dtype=np.float32) + np.array([2.0 * np.pi], dtype=np.float32) * lhs(1, 251)
    x_cyld = r * np.cos(theta) + 0.2
    y_cyld = r * np.sin(theta) + 0.2
    cyld = np.concatenate((x_cyld, y_cyld), axis=1)

    wall = np.concatenate((cyld, wall_up, wall_lw), axis=0)

    xy_c = lb + (ub - lb) * lhs(2, n_collo)
    xy_c_refine = np.array([0.1, 0.1], dtype=np.float32) + np.array([0.2, 0.2], dtype=np.float32) * lhs(2, n_refine)
    xy_c = np.concatenate((xy_c, xy_c_refine), axis=0)
    xy_c = del_cyl_pt(xy_c, xc=0.2, yc=0.2, r=0.05)
    xy_c = np.concatenate((xy_c, wall, cyld, outlet, inlet[:, 0:2]), axis=0)

    return {
        "lb": lb,
        "ub": ub,
        "x_c": to_tensor(xy_c[:, 0:1], requires_grad=True, device=device),
        "y_c": to_tensor(xy_c[:, 1:2], requires_grad=True, device=device),
        "x_wall": to_tensor(wall[:, 0:1], requires_grad=True, device=device),
        "y_wall": to_tensor(wall[:, 1:2], requires_grad=True, device=device),
        "x_inlet": to_tensor(inlet[:, 0:1], requires_grad=True, device=device),
        "y_inlet": to_tensor(inlet[:, 1:2], requires_grad=True, device=device),
        "u_inlet": to_tensor(inlet[:, 2:3], device=device),
        "v_inlet": to_tensor(inlet[:, 3:4], device=device),
        "x_outlet": to_tensor(outlet[:, 0:1], requires_grad=True, device=device),
        "y_outlet": to_tensor(outlet[:, 1:2], requires_grad=True, device=device),
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
                "Adam %d | total=%.3e f=%.3e wall=%.3e inlet=%.3e outlet=%.3e lr=%.3e"
                % (
                    iteration,
                    snapshot["loss"],
                    snapshot["loss_f"],
                    snapshot["loss_wall"],
                    snapshot["loss_inlet"],
                    snapshot["loss_outlet"],
                    optimizer.param_groups[0]["lr"],
                )
            , flush=True)

        # Early stop is based on total training loss from Adam stage.
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
                    "LBFGS %d | total=%.3e f=%.3e wall=%.3e inlet=%.3e outlet=%.3e"
                    % (
                        step_counter["count"],
                        losses["loss"].detach().cpu().item(),
                        losses["loss_f"].detach().cpu().item(),
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
    parser = argparse.ArgumentParser(description="Steady mixed-form PINN in PyTorch")
    parser.add_argument("--adam-iters", type=int, default=30000)
    parser.add_argument("--lbfgs-iters", type=int, default=0)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-collo", type=int, default=40000)
    parser.add_argument("--n-refine", type=int, default=10000)
    parser.add_argument("--checkpoint", default="uvNN_torch.pt")
    parser.add_argument("--best-checkpoint", default="uvNN_torch_best.pt")
    parser.add_argument("--load-checkpoint", default="")
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--save-every", type=int, default=0)
    parser.add_argument("--save-best", action="store_true")
    parser.add_argument("--output-figure", default="uvp_torch.png")
    parser.add_argument("--loss-history", default="loss_history_torch.pickle")
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda", "mps"])
    parser.add_argument("--print-every", type=int, default=100)
    parser.add_argument("--scheduler", default="plateau", choices=["none", "cosine", "step", "plateau"])
    parser.add_argument("--scheduler-step-size", type=int, default=1000)
    parser.add_argument("--scheduler-gamma", type=float, default=0.5)
    parser.add_argument("--scheduler-min-lr", type=float, default=1e-6)
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

    uv_layers = [2] + 8 * [40] + [5]
    data = build_training_data(device=device, n_collo=args.n_collo, n_refine=args.n_refine)
    model = PINNLaminarFlow(uv_layers=uv_layers, lb=data["lb"], ub=data["ub"]).to(device)

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
            "n_collo": args.n_collo,
            "n_refine": args.n_refine,
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

    with open(args.loss_history, "wb") as handle:
        pickle.dump(history, handle)

    base_dir = os.path.dirname(os.path.abspath(__file__))
    ref_path = os.path.abspath(os.path.join(base_dir, "..", "FluentReferenceMu002", "FluentSol.mat"))
    x_fluent, y_fluent, u_fluent, v_fluent, p_fluent = preprocess_reference(ref_path)
    field_fluent = [x_fluent, y_fluent, u_fluent, v_fluent, p_fluent]

    x_pinn = np.linspace(0.0, 1.1, 251)
    y_pinn = np.linspace(0.0, 0.41, 101)
    x_pinn, y_pinn = np.meshgrid(x_pinn, y_pinn)
    x_pinn = x_pinn.flatten()[:, None]
    y_pinn = y_pinn.flatten()[:, None]
    dst = np.sqrt((x_pinn - 0.2) ** 2 + (y_pinn - 0.2) ** 2)
    x_pinn = x_pinn[dst >= 0.05].reshape(-1, 1)
    y_pinn = y_pinn[dst >= 0.05].reshape(-1, 1)

    u_pinn, v_pinn, p_pinn = model.predict(x_pinn, y_pinn)
    field_pinn = [x_pinn, y_pinn, u_pinn, v_pinn, p_pinn]
    post_process(
        xmin=0.0,
        xmax=1.1,
        ymin=0.0,
        ymax=0.41,
        field_fluent=field_fluent,
        field_pinn=field_pinn,
        output_path=args.output_figure,
        s=3,
        alpha=0.5,
    )


if __name__ == "__main__":
    main()
