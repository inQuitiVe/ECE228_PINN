import os
import random
import shutil
import time

# Set GPU visibility before importing TensorFlow.
os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401
from scipy.optimize import minimize


PINN_TF_DEVICE = os.environ.get("PINN_TF_DEVICE", "/GPU:0")
COLLOCATION_FRACTION = float(os.environ.get("PINN_COLLOCATION_FRACTION", "0.25"))
BOUNDARY_FRACTION = float(os.environ.get("PINN_BOUNDARY_FRACTION", "1.0"))
ADAM_ITERS = int(os.environ.get("PINN_ADAM_ITERS", "5000"))
LBFGS_MAXITER = int(os.environ.get("PINN_LBFGS_MAXITER", "100000"))
LBFGS_PRINT_EVERY = int(os.environ.get("PINN_LBFGS_PRINT_EVERY", "10"))
LEARNING_RATE = float(os.environ.get("PINN_LEARNING_RATE", "5e-4"))
WRITE_FRAMES = int(os.environ.get("PINN_WRITE_FRAMES", "0"))


random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)
tf.config.set_soft_device_placement(True)
tf.config.run_functions_eagerly(True)
for gpu in tf.config.list_physical_devices("GPU"):
    try:
        tf.config.experimental.set_memory_growth(gpu, True)
    except RuntimeError:
        pass


def scaled_count(count):
    return max(1, int(round(count * COLLOCATION_FRACTION)))


def lhs(dimensions, samples):
    """Small local Latin-hypercube sampler to avoid pyDOE on Kaggle."""
    result = np.empty((samples, dimensions), dtype=np.float32)
    for dim in range(dimensions):
        perm = np.random.permutation(samples)
        result[:, dim] = (perm + np.random.rand(samples)) / samples
    return result


def subsample_rows(array, fraction, min_rows=1):
    """Subsample fixed boundary/IC arrays for smoke tests only."""
    if fraction >= 1.0:
        return array
    keep = max(min_rows, int(round(array.shape[0] * fraction)))
    keep = min(keep, array.shape[0])
    indices = np.sort(np.random.choice(array.shape[0], size=keep, replace=False))
    return array[indices]


def grad(tape, y, x):
    value = tape.gradient(y, x)
    if value is None:
        return tf.zeros_like(x)
    return value


class MLP(tf.keras.Model):
    def __init__(self, layers):
        super().__init__()
        self.hidden = []
        for in_dim, out_dim in zip(layers[:-2], layers[1:-1]):
            std = np.sqrt(2.0 / (in_dim + out_dim))
            self.hidden.append(
                tf.keras.layers.Dense(
                    out_dim,
                    activation=tf.nn.tanh,
                    kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=std),
                    bias_initializer="zeros",
                    dtype=tf.float32,
                )
            )
        std = np.sqrt(2.0 / (layers[-2] + layers[-1]))
        self.out = tf.keras.layers.Dense(
            layers[-1],
            activation=None,
            kernel_initializer=tf.keras.initializers.TruncatedNormal(stddev=std),
            bias_initializer="zeros",
            dtype=tf.float32,
        )

    def call(self, inputs):
        h = inputs
        for layer in self.hidden:
            h = layer(h)
        return self.out(h)


class PINNLaminarFlowTransient(tf.keras.Model):
    def __init__(self, uv_layers, lb, ub, rho=1.0, mu=0.005):
        super().__init__()
        self.rho = tf.constant(rho, dtype=tf.float32)
        self.mu = tf.constant(mu, dtype=tf.float32)
        self.lb = tf.reshape(tf.convert_to_tensor(lb, dtype=tf.float32), (1, -1))
        self.ub = tf.reshape(tf.convert_to_tensor(ub, dtype=tf.float32), (1, -1))
        self.network = MLP(uv_layers)

    def call(self, inputs):
        scaled = 2.0 * (inputs - self.lb) / (self.ub - self.lb) - 1.0
        return self.network(scaled)

    def net_uv(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            outputs = self(tf.concat([x, y, t], axis=1))
            psi = outputs[:, 0:1]
            p = outputs[:, 1:2]
            s11 = outputs[:, 2:3]
            s22 = outputs[:, 3:4]
            s12 = outputs[:, 4:5]
        u = grad(tape, psi, y)
        v = -grad(tape, psi, x)
        del tape
        return u, v, p, s11, s22, s12

    def net_f(self, x, y, t):
        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y, t])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y])
                outputs = self(tf.concat([x, y, t], axis=1))
                psi = outputs[:, 0:1]
                p = outputs[:, 1:2]
                s11 = outputs[:, 2:3]
                s22 = outputs[:, 3:4]
                s12 = outputs[:, 4:5]
            u = grad(tape1, psi, y)
            v = -grad(tape1, psi, x)

        s11_x = grad(tape2, s11, x)
        s12_y = grad(tape2, s12, y)
        s22_y = grad(tape2, s22, y)
        s12_x = grad(tape2, s12, x)

        u_x = grad(tape2, u, x)
        u_y = grad(tape2, u, y)
        v_x = grad(tape2, v, x)
        v_y = grad(tape2, v, y)
        u_t = grad(tape2, u, t)
        v_t = grad(tape2, v, t)

        del tape1
        del tape2

        f_u = self.rho * u_t + self.rho * (u * u_x + v * u_y) - s11_x - s12_y
        f_v = self.rho * v_t + self.rho * (u * v_x + v * v_y) - s12_x - s22_y
        f_s11 = -p + 2.0 * self.mu * u_x - s11
        f_s22 = -p + 2.0 * self.mu * v_y - s22
        f_s12 = self.mu * (u_y + v_x) - s12
        f_p = p + (s11 + s22) / 2.0
        return f_u, f_v, f_s11, f_s22, f_s12, f_p

    def predict(self, x, y, t):
        x_t = tf.convert_to_tensor(np.asarray(x).reshape(-1, 1), dtype=tf.float32)
        y_t = tf.convert_to_tensor(np.asarray(y).reshape(-1, 1), dtype=tf.float32)
        t_t = tf.convert_to_tensor(np.asarray(t).reshape(-1, 1), dtype=tf.float32)
        u, v, p, _, _, _ = self.net_uv(x_t, y_t, t_t)
        return u.numpy(), v.numpy(), p.numpy()


def to_tensor(array):
    return tf.convert_to_tensor(array, dtype=tf.float32)


def mse_zero(x):
    return tf.reduce_mean(tf.square(x))


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
    theta = np.linspace(0.0, 2.0 * np.pi, num_r)
    x = r * np.cos(theta) + xc
    y = r * np.sin(theta) + yc
    t = np.linspace(tmin, tmax, num_t)
    xx, tt = np.meshgrid(x, t)
    yy, _ = np.meshgrid(y, t)
    return xx.flatten()[:, None], yy.flatten()[:, None], tt.flatten()[:, None]


def build_training_data(tmax=0.5, period=1.0):
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
    x_inb, y_inb, t_inb = cart_grid(0, 0, 0, 0.41, 0, tmax, 1, 61, 61)
    u_inb = 4.0 * u_max * y_inb * (0.41 - y_inb) / (0.41 ** 2) * (
        np.sin(2.0 * np.pi * t_inb / period + 1.5 * np.pi) + 1.0
    )
    v_inb = np.zeros_like(x_inb)
    inb = np.concatenate((x_inb, y_inb, t_inb, u_inb, v_inb), axis=1)

    x_outb, y_outb, t_outb = cart_grid(1.1, 1.1, 0, 0.41, 0, tmax, 1, 81, 41)
    outb = np.concatenate((x_outb, y_outb, t_outb), axis=1)

    r = 0.05
    x_surf, y_surf, t_surf = gen_circle_pt(0.2, 0.2, r, 0, tmax, 81, 51)
    hole = np.concatenate((x_surf, y_surf, t_surf), axis=1)
    wall = np.concatenate((hole, wall_up, wall_lw), axis=0)

    if BOUNDARY_FRACTION < 1.0:
        print("BOUNDARY_FRACTION:", BOUNDARY_FRACTION, flush=True)
        print(
            "Boundary rows before subsample ic/wall/inlet/outlet:",
            ic.shape[0], wall.shape[0], inb.shape[0], outb.shape[0],
            flush=True,
        )
        ic = subsample_rows(ic, BOUNDARY_FRACTION, min_rows=64)
        wall = subsample_rows(wall, BOUNDARY_FRACTION, min_rows=64)
        inb = subsample_rows(inb, BOUNDARY_FRACTION, min_rows=64)
        outb = subsample_rows(outb, BOUNDARY_FRACTION, min_rows=64)
        print(
            "Boundary rows after subsample ic/wall/inlet/outlet:",
            ic.shape[0], wall.shape[0], inb.shape[0], outb.shape[0],
            flush=True,
        )

    n_base = scaled_count(80000)
    n_refine = scaled_count(15000)
    n_wall = scaled_count(3000)
    print("COLLOCATION_FRACTION:", COLLOCATION_FRACTION, flush=True)
    print("LHS counts base/refine/lower/upper:", n_base, n_refine, n_wall, n_wall, flush=True)

    xy_c = lb + (ub - lb) * lhs(3, n_base)
    xy_c_refine = np.array([0.0, 0.0, 0.0], dtype=np.float32) + np.array([0.4, 0.4, tmax], dtype=np.float32) * lhs(3, n_refine)
    xy_c_lw = np.array([0.0, 0.0, 0.0], dtype=np.float32) + np.array([1.1, 0.02, tmax], dtype=np.float32) * lhs(3, n_wall)
    xy_c_up = np.array([0.0, 0.39, 0.0], dtype=np.float32) + np.array([1.1, 0.02, tmax], dtype=np.float32) * lhs(3, n_wall)
    xy_c = np.concatenate((xy_c, xy_c_refine, xy_c_lw, xy_c_up), axis=0)
    xy_c = del_src_pt(xy_c, xc=0.2, yc=0.2, r=0.05)
    xy_c = np.concatenate((xy_c, wall, outb, inb[:, 0:3]), axis=0)
    print("Final collocation shape:", xy_c.shape, flush=True)

    save_inlet_profile(inb)

    return {
        "lb": lb,
        "ub": ub,
        "x_c": to_tensor(xy_c[:, 0:1]),
        "y_c": to_tensor(xy_c[:, 1:2]),
        "t_c": to_tensor(xy_c[:, 2:3]),
        "x_ic": to_tensor(ic[:, 0:1]),
        "y_ic": to_tensor(ic[:, 1:2]),
        "t_ic": to_tensor(ic[:, 2:3]),
        "x_wall": to_tensor(wall[:, 0:1]),
        "y_wall": to_tensor(wall[:, 1:2]),
        "t_wall": to_tensor(wall[:, 2:3]),
        "x_inlet": to_tensor(inb[:, 0:1]),
        "y_inlet": to_tensor(inb[:, 1:2]),
        "t_inlet": to_tensor(inb[:, 2:3]),
        "u_inlet": to_tensor(inb[:, 3:4]),
        "v_inlet": to_tensor(inb[:, 4:5]),
        "x_outlet": to_tensor(outb[:, 0:1]),
        "y_outlet": to_tensor(outb[:, 1:2]),
        "t_outlet": to_tensor(outb[:, 2:3]),
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
    loss_inlet = tf.reduce_mean(tf.square(u_inlet - data["u_inlet"])) + tf.reduce_mean(tf.square(v_inlet - data["v_inlet"]))
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


def loss_to_numpy(losses):
    return {name: float(value.numpy()) for name, value in losses.items()}


def print_losses(label, losses):
    values = loss_to_numpy(losses)
    print(
        "%s | total=%.6e f=%.6e ic=%.6e wall=%.6e inlet=%.6e outlet=%.6e"
        % (
            label,
            values["loss"],
            values["loss_f"],
            values["loss_ic"],
            values["loss_wall"],
            values["loss_inlet"],
            values["loss_outlet"],
        ),
        flush=True,
    )


def adam_step(model, data, optimizer):
    with tf.GradientTape() as tape:
        losses = build_loss(model, data)
    grads = tape.gradient(losses["loss"], model.trainable_variables)
    optimizer.apply_gradients(zip(grads, model.trainable_variables))
    return losses


def train_adam(model, data, iters, learning_rate):
    optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)
    for iteration in range(iters):
        losses = adam_step(model, data, optimizer)
        if iteration == 0 or iteration % 100 == 0 or iteration == iters - 1:
            print_losses("Adam %d" % iteration, losses)


def flatten_parameters(model):
    return np.concatenate([var.numpy().ravel() for var in model.trainable_variables]).astype(np.float64)


def assign_flat_parameters(model, flat_params):
    offset = 0
    for var in model.trainable_variables:
        size = int(np.prod(var.shape))
        values = flat_params[offset:offset + size].reshape(var.shape)
        var.assign(tf.convert_to_tensor(values, dtype=var.dtype))
        offset += size


def flatten_gradients(grads, variables):
    flat = []
    for grad_value, var in zip(grads, variables):
        if grad_value is None:
            flat.append(np.zeros(var.shape, dtype=np.float32).ravel())
        else:
            flat.append(grad_value.numpy().ravel())
    return np.concatenate(flat).astype(np.float64)


def train_lbfgs(model, data, maxiter):
    state = {"evals": 0, "best": np.inf}

    def objective(flat_params):
        assign_flat_parameters(model, flat_params)
        with tf.GradientTape() as tape:
            losses = build_loss(model, data)
        grads = tape.gradient(losses["loss"], model.trainable_variables)
        loss_value = float(losses["loss"].numpy())
        grad_value = flatten_gradients(grads, model.trainable_variables)
        state["evals"] += 1
        state["best"] = min(state["best"], loss_value)
        if state["evals"] == 1 or state["evals"] % LBFGS_PRINT_EVERY == 0:
            print_losses("L-BFGS %d" % state["evals"], losses)
        return loss_value, grad_value

    print_losses("Before L-BFGS", build_loss(model, data))
    result = minimize(
        fun=objective,
        x0=flatten_parameters(model),
        jac=True,
        method="L-BFGS-B",
        options={
            "maxiter": maxiter,
            "maxfun": maxiter,
            "maxcor": 50,
            "maxls": 50,
            "ftol": np.finfo(float).eps,
            "gtol": 0.0,
        },
    )
    assign_flat_parameters(model, result.x)
    print_losses("After L-BFGS", build_loss(model, data))
    print("SciPy L-BFGS-B evals:", state["evals"], flush=True)
    print("SciPy L-BFGS-B best loss: %.6e" % state["best"], flush=True)
    print("SciPy L-BFGS-B status:", result.message, flush=True)


def save_inlet_profile(inb):
    os.makedirs("./output", exist_ok=True)
    fig = plt.figure()
    ax = fig.add_subplot(111, projection="3d")
    ax.scatter(inb[:, 1:2], inb[:, 2:3], inb[:, 3:4], marker="o", alpha=0.1, s=2, color="blue")
    ax.set_xlabel("y axis")
    ax.set_ylabel("t axis")
    ax.set_zlabel("u axis")
    plt.savefig("./output/inlet_profile_tf2_debug.png", dpi=150)
    plt.close(fig)


def save_front_pressure(model, tmax):
    os.makedirs("./output", exist_ok=True)
    t_front = np.linspace(0, tmax, 100).reshape(-1, 1)
    x_front = np.full_like(t_front, 0.15)
    y_front = np.full_like(t_front, 0.20)
    _, _, p_front = model.predict(x_front, y_front, t_front)
    plt.figure()
    plt.plot(t_front, p_front)
    plt.title("Pressure at leading point")
    plt.xlabel("t")
    plt.ylabel("p")
    plt.savefig("./output/front_pressure_tf2_debug.png", dpi=150)
    plt.close()


def post_process(xmin, xmax, ymin, ymax, field, s=2, num=0):
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
    plt.suptitle("Time: " + str(num * 0.01) + "s", fontsize=16)
    plt.savefig("./output/uvp_comparison_tf2_" + str(num) + ".png", dpi=150)
    plt.close(fig)


def write_prediction_frames(model):
    n_t = 51
    x_star = np.linspace(0, 1.1, 401)
    y_star = np.linspace(0, 0.41, 161)
    x_star, y_star = np.meshgrid(x_star, y_star)
    x_star = x_star.flatten()[:, None]
    y_star = y_star.flatten()[:, None]
    dst = np.sqrt((x_star - 0.2) ** 2 + (y_star - 0.2) ** 2)
    x_star = x_star[dst >= 0.05].reshape(-1, 1)
    y_star = y_star[dst >= 0.05].reshape(-1, 1)
    for i in range(n_t):
        t_star = np.full((x_star.size, 1), i * 0.5 / (n_t - 1))
        u_pred, v_pred, p_pred = model.predict(x_star, y_star, t_star)
        field = [x_star, y_star, t_star, u_pred, v_pred, p_pred]
        post_process(xmin=0, xmax=1.1, ymin=0, ymax=0.41, field=field, s=2, num=i)


def main():
    print("TensorFlow:", tf.__version__, flush=True)
    print("CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"), flush=True)
    print("Physical GPUs:", tf.config.list_physical_devices("GPU"), flush=True)
    print("Using TensorFlow device scope:", PINN_TF_DEVICE, flush=True)
    print("ADAM_ITERS:", ADAM_ITERS, "LBFGS_MAXITER:", LBFGS_MAXITER, flush=True)

    shutil.rmtree("./output", ignore_errors=True)
    os.makedirs("./output", exist_ok=True)

    tmax = 0.5
    uv_layers = [3] + 7 * [50] + [5]
    with tf.device(PINN_TF_DEVICE):
        data = build_training_data(tmax=tmax, period=tmax * 2.0)
        model = PINNLaminarFlowTransient(uv_layers=uv_layers, lb=data["lb"], ub=data["ub"])
        _ = build_loss(model, data)

        start_time = time.time()
        train_adam(model, data, iters=ADAM_ITERS, learning_rate=LEARNING_RATE)
        print_losses("After Adam", build_loss(model, data))
        train_lbfgs(model, data, maxiter=LBFGS_MAXITER)
        save_front_pressure(model, tmax)
        if WRITE_FRAMES:
            write_prediction_frames(model)
        print("--- %.2f seconds ---" % (time.time() - start_time), flush=True)


if __name__ == "__main__":
    main()
