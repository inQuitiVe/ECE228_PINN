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
    def __init__(self, uv_layers, rho=1.0, mu=0.005):
        super().__init__()
        self.rho = rho
        self.mu = mu
        self.network = MLP(uv_layers)

    def net_uv(self, x, y, t):
        outputs = self.network(torch.cat([x, y, t], dim=1))
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


def train(model, data, adam_iters, learning_rate, lbfgs_steps, print_every):
    history = []
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)
    start_time = time.time()

    model.train()
    for iteration in range(1, adam_iters + 1):
        optimizer.zero_grad(set_to_none=True)
        losses = build_loss(model, data)
        losses["loss"].backward()
        optimizer.step()

        snapshot = {name: value.detach().cpu().item() for name, value in losses.items()}
        history.append(snapshot)
        if iteration == 1 or iteration % print_every == 0:
            print(
                "Adam %d | total=%.3e f=%.3e ic=%.3e wall=%.3e inlet=%.3e outlet=%.3e"
                % (
                    iteration,
                    snapshot["loss"],
                    snapshot["loss_f"],
                    snapshot["loss_ic"],
                    snapshot["loss_wall"],
                    snapshot["loss_inlet"],
                    snapshot["loss_outlet"],
                )
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
                )
            return losses["loss"]

        lbfgs.step(closure)
        final_losses = build_loss(model, data)
        history.append({name: value.detach().cpu().item() for name, value in final_losses.items()})

    print("--- %.2f seconds ---" % (time.time() - start_time))
    return history


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


def save_checkpoint(model, path, history, config):
    torch.save(
        {
            "model_state_dict": model.state_dict(),
            "history": history,
            "config": config,
        },
        path,
    )
    print("Saved PyTorch checkpoint to %s" % path)


def load_checkpoint(model, path, map_location):
    checkpoint = torch.load(path, map_location=map_location)
    model.load_state_dict(checkpoint["model_state_dict"])
    print("Loaded PyTorch checkpoint from %s" % path)


def parse_args():
    parser = argparse.ArgumentParser(description="Transient mixed-form PINN in PyTorch")
    parser.add_argument("--adam-iters", type=int, default=5000)
    parser.add_argument("--lbfgs-iters", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--tmax", type=float, default=0.5)
    parser.add_argument("--checkpoint", default="uvNN_torch.pt")
    parser.add_argument("--load-checkpoint", default="")
    parser.add_argument("--output-dir", default="output")
    parser.add_argument("--loss-history", default="loss_history_torch.pickle")
    parser.add_argument("--num-frames", type=int, default=51)
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print("Using device:", device)

    torch.manual_seed(1234)
    np.random.seed(1234)

    uv_layers = [3] + 7 * [50] + [5]
    data = build_training_data(device=device, tmax=args.tmax)
    model = PINNLaminarFlowTransient(uv_layers=uv_layers).to(device)

    if args.load_checkpoint:
        load_checkpoint(model, args.load_checkpoint, map_location=device)

    history = train(
        model=model,
        data=data,
        adam_iters=args.adam_iters,
        learning_rate=args.learning_rate,
        lbfgs_steps=args.lbfgs_iters,
        print_every=args.print_every,
    )

    save_checkpoint(
        model,
        args.checkpoint,
        history,
        {
            "uv_layers": uv_layers,
            "adam_iters": args.adam_iters,
            "lbfgs_iters": args.lbfgs_iters,
            "learning_rate": args.learning_rate,
            "tmax": args.tmax,
        },
    )

    with open(args.loss_history, "wb") as handle:
        pickle.dump(history, handle)

    t_front = np.linspace(0, args.tmax, 100).reshape(-1, 1)
    x_front = np.full_like(t_front, 0.15)
    y_front = np.full_like(t_front, 0.20)
    _, _, p_front = model.predict(x_front, y_front, t_front)
    plt.figure()
    plt.plot(t_front, p_front)
    plt.title("Pressure at leading point")
    plt.xlabel("t")
    plt.ylabel("p")
    plt.savefig("pressure_front_torch.png", dpi=150)
    plt.close()

    x_star = np.linspace(0, 1.1, 401)
    y_star = np.linspace(0, 0.41, 161)
    x_star, y_star = np.meshgrid(x_star, y_star)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]
    dst = np.sqrt((x_star - 0.2) ** 2 + (y_star - 0.2) ** 2)
    x_star = x_star[dst >= 0.05].reshape(-1, 1)
    y_star = y_star[dst >= 0.05].reshape(-1, 1)

    shutil.rmtree(args.output_dir, ignore_errors=True)
    os.makedirs(args.output_dir, exist_ok=True)
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
            out_path=os.path.join(args.output_dir, "uvp_comparison_%d.png" % i),
            s=2,
            title="Time: %.3fs" % (i * args.tmax / (args.num_frames - 1)),
        )


if __name__ == "__main__":
    main()
