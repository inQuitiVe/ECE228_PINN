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
    def __init__(self, uv_layers, rho=1.0, mu=0.02):
        super().__init__()
        self.rho = rho
        self.mu = mu
        self.network = MLP(uv_layers)

    def net_uv(self, x, y):
        outputs = self.network(torch.cat([x, y], dim=1))
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
                "Adam %d | total=%.3e f=%.3e wall=%.3e inlet=%.3e outlet=%.3e"
                % (
                    iteration,
                    snapshot["loss"],
                    snapshot["loss_f"],
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
                    "LBFGS %d | total=%.3e f=%.3e wall=%.3e inlet=%.3e outlet=%.3e"
                    % (
                        step_counter["count"],
                        losses["loss"].detach().cpu().item(),
                        losses["loss_f"].detach().cpu().item(),
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
    return checkpoint


def parse_args():
    parser = argparse.ArgumentParser(description="Steady mixed-form PINN in PyTorch")
    parser.add_argument("--adam-iters", type=int, default=10000)
    parser.add_argument("--lbfgs-iters", type=int, default=500)
    parser.add_argument("--learning-rate", type=float, default=5e-4)
    parser.add_argument("--n-collo", type=int, default=40000)
    parser.add_argument("--n-refine", type=int, default=10000)
    parser.add_argument("--checkpoint", default="uvNN_torch.pt")
    parser.add_argument("--load-checkpoint", default="")
    parser.add_argument("--output-figure", default="uvp_torch.png")
    parser.add_argument("--loss-history", default="loss_history_torch.pickle")
    parser.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--print-every", type=int, default=100)
    return parser.parse_args()


def main():
    args = parse_args()
    device = torch.device(args.device)
    print("Using device:", device)

    torch.manual_seed(1234)
    np.random.seed(1234)

    uv_layers = [2] + 8 * [40] + [5]
    data = build_training_data(device=device, n_collo=args.n_collo, n_refine=args.n_refine)
    model = PINNLaminarFlow(uv_layers=uv_layers).to(device)

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
            "n_collo": args.n_collo,
            "n_refine": args.n_refine,
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
