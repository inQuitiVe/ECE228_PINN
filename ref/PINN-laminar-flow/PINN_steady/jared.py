import numpy as np
import time
from pyDOE import lhs
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
import pickle
import scipy.io
import random

import tensorflow as tf
import os

# TF2: suppress verbose logs
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# GPU selection: set visible devices before any other TF calls
# CPU: set to '' or comment out; GPU0: '0'; GPU1: '1'
os.environ['CUDA_VISIBLE_DEVICES'] = '0'

random.seed(1234)
np.random.seed(1234)
tf.random.set_seed(1234)


class PINN_laminar_flow:
    """
    Mixed-variable PINN for steady incompressible laminar flow.
    Migrated from TensorFlow v1 to TensorFlow v2 (eager + tf.function).

    Key changes vs TF1:
    - No more tf.Session / tf.placeholder / tf.Variable graph building.
    - Weights/biases are plain Python lists of tf.Variable.
    - Forward pass and loss are computed inside tf.GradientTape.
    - Adam optimiser uses tf.keras.optimizers.Adam.
    - L-BFGS minimisation uses scipy.optimize.minimize with a flat-variable
      packing/unpacking helper (TF2 no longer ships ScipyOptimizerInterface).
    - neural_net input normalisation is done with tf operations on tensors.
    """

    def __init__(self, Collo, INLET, OUTLET, WALL, uv_layers, lb, ub,
                 ExistModel=0, uvDir=''):

        self.count = 0

        # Bounds (stored as float32 tensors for use in neural_net)
        self.lb = tf.constant(lb, dtype=tf.float32)
        self.ub = tf.constant(ub, dtype=tf.float32)

        # Material properties
        self.rho = 1.0
        self.mu  = 0.02

        # ---- Training data (kept as numpy, cast to tensor on-the-fly) ----
        self.x_c = Collo[:, 0:1].astype(np.float32)
        self.y_c = Collo[:, 1:2].astype(np.float32)

        self.x_INLET = INLET[:, 0:1].astype(np.float32)
        self.y_INLET = INLET[:, 1:2].astype(np.float32)
        self.u_INLET = INLET[:, 2:3].astype(np.float32)
        self.v_INLET = INLET[:, 3:4].astype(np.float32)

        self.x_OUTLET = OUTLET[:, 0:1].astype(np.float32)
        self.y_OUTLET = OUTLET[:, 1:2].astype(np.float32)

        self.x_WALL = WALL[:, 0:1].astype(np.float32)
        self.y_WALL = WALL[:, 1:2].astype(np.float32)

        # Network layout
        self.uv_layers = uv_layers

        # Loss history (replaces loss_rec from TF1 version)
        self.loss_rec = []

        # Initialise (or load) weights and biases
        if ExistModel == 0:
            self.uv_weights, self.uv_biases = self.initialize_NN(uv_layers)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, uv_layers)

        # Collect all trainable variables in one flat list for the optimisers
        self.trainable_variables = self.uv_weights + self.uv_biases

    # ------------------------------------------------------------------
    # Network initialisation
    # ------------------------------------------------------------------

    def initialize_NN(self, layers):
        weights = []
        biases  = []
        for l in range(len(layers) - 1):
            W = self.xavier_init([layers[l], layers[l + 1]])
            b = tf.Variable(tf.zeros([1, layers[l + 1]], dtype=tf.float32))
            weights.append(W)
            biases.append(b)
        return weights, biases

    def xavier_init(self, size):
        in_dim, out_dim = size
        std = np.sqrt(2.0 / (in_dim + out_dim))
        return tf.Variable(
            tf.random.truncated_normal([in_dim, out_dim], stddev=std,
                                       dtype=tf.float32))

    # ------------------------------------------------------------------
    # Save / load
    # ------------------------------------------------------------------

    def save_NN(self, fileDir):
        uv_w = [w.numpy() for w in self.uv_weights]
        uv_b = [b.numpy() for b in self.uv_biases]
        with open(fileDir, 'wb') as f:
            pickle.dump([uv_w, uv_b], f)
        print("Save uv NN parameters successfully...")

    def load_NN(self, fileDir, layers):
        weights, biases = [], []
        with open(fileDir, 'rb') as f:
            uv_w, uv_b = pickle.load(f)
        assert len(layers) == len(uv_w) + 1
        for w_arr, b_arr in zip(uv_w, uv_b):
            weights.append(tf.Variable(w_arr, dtype=tf.float32))
            biases.append(tf.Variable(b_arr, dtype=tf.float32))
            print(" - Load NN parameters successfully...")
        return weights, biases

    # ------------------------------------------------------------------
    # Forward pass
    # ------------------------------------------------------------------

    def neural_net(self, X):
        """
        Fully-connected tanh network with input normalisation to [-1, 1].
        NOTE: In TF1 version the steady case had normalisation commented out.
              Kept consistent here — normalisation IS applied (matches TF1
              transient case and the paper). Toggle by commenting the H line.
        """
        # Input normalisation to [-1, 1] (recommended; matches the paper and
        # the TF2 transient case). Linear map, so gradients flow unchanged.
        H = 2.0 * (X - self.lb) / (self.ub - self.lb) - 1.0

        for W, b in zip(self.uv_weights[:-1], self.uv_biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        Y = tf.matmul(H, self.uv_weights[-1]) + self.uv_biases[-1]
        return Y

    def net_uv(self, x, y):
        """
        Returns u, v, p, s11, s22, s12.
        Stream-function trick enforces div(v)=0 automatically.
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            xy    = tf.concat([x, y], axis=1)
            out   = self.neural_net(xy)
            psi   = out[:, 0:1]
            p     = out[:, 1:2]
            s11   = out[:, 2:3]
            s22   = out[:, 3:4]
            s12   = out[:, 4:5]
        u =  tape.gradient(psi, y)
        v = -tape.gradient(psi, x)
        del tape
        return u, v, p, s11, s22, s12

    def net_f(self, x, y):
        """PDE residuals for the mixed-variable formulation."""
        rho, mu = self.rho, self.mu

        with tf.GradientTape(persistent=True) as tape2:
            tape2.watch([x, y])
            with tf.GradientTape(persistent=True) as tape1:
                tape1.watch([x, y])
                xy   = tf.concat([x, y], axis=1)
                out  = self.neural_net(xy)
                psi  = out[:, 0:1]
                p    = out[:, 1:2]
                s11  = out[:, 2:3]
                s22  = out[:, 3:4]
                s12  = out[:, 4:5]
            u   =  tape1.gradient(psi, y)
            v   = -tape1.gradient(psi, x)
            # Second derivatives must use the OUTER tape: u and v were produced
            # by tape1.gradient(...) inside tape2's context, so only tape2 has
            # recorded how they depend on x, y. Using tape1 here returns None.
            u_x = tape2.gradient(u, x)
            u_y = tape2.gradient(u, y)
            v_x = tape2.gradient(v, x)
            v_y = tape2.gradient(v, y)
            del tape1

        s11_x = tape2.gradient(s11, x)
        s12_y = tape2.gradient(s12, y)
        s22_y = tape2.gradient(s22, y)
        s12_x = tape2.gradient(s12, x)
        del tape2

        f_u   = rho * (u * u_x + v * u_y) - s11_x - s12_y
        f_v   = rho * (u * v_x + v * v_y) - s12_x - s22_y

        f_s11 = -p + 2 * mu * u_x - s11
        f_s22 = -p + 2 * mu * v_y - s22
        f_s12 = mu * (u_y + v_x) - s12
        f_p   = p + (s11 + s22) / 2.0

        return f_u, f_v, f_s11, f_s22, f_s12, f_p
            
    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------


    def compute_loss(self):
        x_c  = tf.constant(self.x_c)
        y_c  = tf.constant(self.y_c)
        x_W  = tf.constant(self.x_WALL);  y_W = tf.constant(self.y_WALL)
        x_I  = tf.constant(self.x_INLET); y_I = tf.constant(self.y_INLET)
        u_I  = tf.constant(self.u_INLET); v_I = tf.constant(self.v_INLET)
        x_O  = tf.constant(self.x_OUTLET);y_O = tf.constant(self.y_OUTLET)

        # PDE residuals
        f_u, f_v, f_s11, f_s22, f_s12, f_p = self.net_f(x_c, y_c)
        loss_f = (tf.reduce_mean(tf.square(f_u))
                + tf.reduce_mean(tf.square(f_v))
                + tf.reduce_mean(tf.square(f_s11))
                + tf.reduce_mean(tf.square(f_s22))
                + tf.reduce_mean(tf.square(f_s12))
                + tf.reduce_mean(tf.square(f_p)))

        # Wall BC (no-slip)
        u_W, v_W, _, _, _, _ = self.net_uv(x_W, y_W)
        loss_WALL = (tf.reduce_mean(tf.square(u_W))
                   + tf.reduce_mean(tf.square(v_W)))

        # Inlet BC
        u_Ip, v_Ip, _, _, _, _ = self.net_uv(x_I, y_I)
        loss_INLET = (tf.reduce_mean(tf.square(u_Ip - u_I))
                    + tf.reduce_mean(tf.square(v_Ip - v_I)))

        # Outlet BC (p = 0)
        _, _, p_O, _, _, _ = self.net_uv(x_O, y_O)
        loss_OUTLET = tf.reduce_mean(tf.square(p_O))

        loss = loss_f + 2.0 * (loss_WALL + loss_INLET + loss_OUTLET)
        return loss, loss_f, loss_WALL, loss_INLET, loss_OUTLET

    # ------------------------------------------------------------------
    # Adam training
    # ------------------------------------------------------------------

    def train(self, iter, learning_rate):
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        loss_WALL_hist   = []
        loss_f_hist      = []
        loss_INLET_hist  = []
        loss_OUTLET_hist = []

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                loss, loss_f, loss_W, loss_I, loss_O = self.compute_loss()
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return loss, loss_f, loss_W, loss_I, loss_O

        for it in range(iter):
            
            loss, loss_f, loss_W, loss_I, loss_O = train_step()

            if it % 10 == 0:
                print(f'It: {it}, Loss: {loss.numpy():.3e}')

            self.loss_rec.append(loss.numpy())
            loss_WALL_hist.append(loss_W.numpy())
            loss_f_hist.append(loss_f.numpy())
            loss_INLET_hist.append(loss_I.numpy())
            loss_OUTLET_hist.append(loss_O.numpy())

        return loss_WALL_hist, loss_INLET_hist, loss_OUTLET_hist, loss_f_hist, self.loss_rec

    # ------------------------------------------------------------------
    # L-BFGS-B training (replaces ScipyOptimizerInterface)
    # ------------------------------------------------------------------

    def _pack_variables(self):
        """Flatten all trainable variables into a single 1-D numpy array."""
        return np.concatenate(
            [v.numpy().flatten() for v in self.trainable_variables])

    def _unpack_variables(self, flat):
        """Write a flat numpy array back into the trainable variables."""
        idx = 0
        for v in self.trainable_variables:
            shape = v.shape
            size  = np.prod(shape)
            v.assign(flat[idx:idx + size].reshape(shape))
            idx  += size

    def _loss_and_grad(self, flat_params):
        """Compute loss + gradients for scipy.optimize.minimize."""
        self._unpack_variables(flat_params)
        with tf.GradientTape() as tape:
            loss, _, _, _, _ = self.compute_loss()
        grads = tape.gradient(loss, self.trainable_variables)
        grad_flat = np.concatenate(
            [g.numpy().flatten() for g in grads])
        self.count += 1
        if self.count % 100 == 0:
            print(f'{self.count} th iterations (L-BFGS), Loss: {loss.numpy():.6e}')
        self.loss_rec.append(loss.numpy())
        return loss.numpy().astype(np.float64), grad_flat.astype(np.float64)

    def train_bfgs(self):
        from scipy.optimize import minimize

        self.iter_count = 0

        def callback(xk):
            # Fired once per real L-BFGS-B iteration (not per function eval).
            # loss_rec[-1] is the most recent loss from _loss_and_grad.
            self.iter_count += 1
            print(f'[L-BFGS] iter {self.iter_count:5d}  '
                  f'loss = {self.loss_rec[-1]:.6e}  (nfev {self.count})')

        print('\n>>> Starting L-BFGS-B optimisation')
        print(f'    Loss before L-BFGS: {self.loss_rec[-1]:.6e}')

        x0 = self._pack_variables().astype(np.float64)
        result = minimize(
            self._loss_and_grad,
            x0,
            method='L-BFGS-B',
            jac=True,
            callback=callback,
            options={
                'maxiter': 100000,
                'maxfun':  100000,
                'maxcor':  50,
                'maxls':   50,
                'ftol':    1.0 * np.finfo(float).eps,
                'gtol':    1e-8,
            }
        )
        self._unpack_variables(result.x)
        print(f'    L-BFGS-B finished: {result.message}')
        print(f'    Loss after  L-BFGS: {self.loss_rec[-1]:.6e}')
        print(f'    Total iters: {self.iter_count}, total nfev: {self.count}')

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, x_star, y_star):
        x = tf.constant(x_star.astype(np.float32))
        y = tf.constant(y_star.astype(np.float32))
        u, v, p, _, _, _ = self.net_uv(x, y)
        return u.numpy(), v.numpy(), p.numpy()

    def getloss(self):
        loss, loss_f, loss_W, loss_I, loss_O = self.compute_loss()
        print(f'loss_f={loss_f.numpy():.3e}, loss_WALL={loss_W.numpy():.3e}, '
              f'loss_INLET={loss_I.numpy():.3e}, loss_OUTLET={loss_O.numpy():.3e}, '
              f'loss={loss.numpy():.3e}')
        return loss_W.numpy(), loss_I.numpy(), loss_O.numpy(), loss_f.numpy(), loss.numpy()


# -----------------------------------------------------------------------
# Helpers (unchanged from TF1 version)
# -----------------------------------------------------------------------

def DelCylPT(XY_c, xc=0.0, yc=0.0, r=0.1):
    """Delete collocation points inside the cylinder."""
    dst = np.array([((xy[0] - xc)**2 + (xy[1] - yc)**2)**0.5 for xy in XY_c])
    return XY_c[dst > r, :]


def postProcess(xmin, xmax, ymin, ymax, field_FLUENT, field_MIXED,
                s=2, alpha=0.5, marker='o'):

    [x_FLUENT, y_FLUENT, u_FLUENT, v_FLUENT, p_FLUENT] = field_FLUENT
    [x_MIXED,  y_MIXED,  u_MIXED,  v_MIXED,  p_MIXED]  = field_MIXED

    fig, ax = plt.subplots(nrows=3, ncols=2, figsize=(7, 4))
    fig.subplots_adjust(hspace=0.2, wspace=0.2)

    titles = [r'$u$ (m/s)', r'$v$ (m/s)', 'Pressure (Pa)']
    data_m = [u_MIXED, v_MIXED, p_MIXED]
    data_f = [u_FLUENT, v_FLUENT, p_FLUENT]
    vmins  = [None, None, -0.25]
    vmaxs  = [None, None,  4.0]

    for row in range(3):
        for col, (data, x, y) in enumerate(
                [(data_m[row], x_MIXED, y_MIXED),
                 (data_f[row], x_FLUENT, y_FLUENT)]):
            cf = ax[row, col].scatter(x, y, c=data, alpha=alpha,
                                      edgecolors='none', cmap='rainbow',
                                      marker=marker, s=int(s),
                                      vmin=vmins[row], vmax=vmaxs[row])
            ax[row, col].axis('square')
            for spine in ax[row, col].spines.values():
                spine.set_visible(False)
            ax[row, col].set_xticks([])
            ax[row, col].set_yticks([])
            ax[row, col].set_xlim([xmin, xmax])
            ax[row, col].set_ylim([ymin, ymax])
            ax[row, col].set_title(titles[row])
            fig.colorbar(cf, ax=ax[row, col], fraction=0.046, pad=0.04)

    plt.savefig('./uvp.png', dpi=300)
    plt.close('all')


def preprocess(dir='FenicsSol.mat'):
    data    = scipy.io.loadmat(dir)
    x_star  = data['x'].flatten()[:, None]
    y_star  = data['y'].flatten()[:, None]
    p_star  = data['p'].flatten()[:, None]
    vx_star = data['vx'].flatten()[:, None]
    vy_star = data['vy'].flatten()[:, None]
    return x_star, y_star, vx_star, vy_star, p_star


# -----------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------

if __name__ == "__main__":

    # Domain bounds
    lb = np.array([0.0, 0.0])
    ub = np.array([1.1,  0.41])

    # Network: 2 inputs, 8 hidden layers × 40 neurons, 5 outputs
    uv_layers = [2] + 8 * [40] + [5]

    # ---- Boundary / collocation points ----

    # WALL (upper + lower channel walls)  [x, y]
    wall_up = np.array([0.0, 0.41]) + np.array([1.1, 0.0]) * lhs(2, 441)
    wall_lw = np.array([0.0, 0.00]) + np.array([1.1, 0.0]) * lhs(2, 441)

    # INLET  [x, y, u, v]
    U_max  = 1.0
    INLET  = np.array([0.0, 0.0]) + np.array([0.0, 0.41]) * lhs(2, 201)
    y_INLET = INLET[:, 1:2]
    u_INLET = 4 * U_max * y_INLET * (0.41 - y_INLET) / (0.41 ** 2)
    v_INLET = np.zeros_like(y_INLET)
    INLET   = np.concatenate((INLET, u_INLET, v_INLET), axis=1)

    # OUTLET [x, y]
    OUTLET = np.array([1.1, 0.0]) + np.array([0.0, 0.41]) * lhs(2, 201)

    # Cylinder surface
    r     = 0.05
    theta = np.array([0.0]) + np.array([2 * np.pi]) * lhs(1, 251)
    x_CYL = np.multiply(r, np.cos(theta)) + 0.2
    y_CYL = np.multiply(r, np.sin(theta)) + 0.2
    CYLD  = np.concatenate((x_CYL, y_CYL), axis=1)

    WALL  = np.concatenate((CYLD, wall_up, wall_lw), axis=0)

    # Collocation points (physics residual)
    XY_c        = lb + (ub - lb) * lhs(2, 40000)
    XY_c_refine = np.array([0.1, 0.1]) + np.array([0.2, 0.2]) * lhs(2, 10000)
    XY_c        = np.concatenate((XY_c, XY_c_refine), axis=0)
    XY_c        = DelCylPT(XY_c, xc=0.2, yc=0.2, r=0.05)
    XY_c        = np.concatenate((XY_c, WALL, CYLD, OUTLET, INLET[:, 0:2]), axis=0)

    print("Total collocation points:", XY_c.shape)

    # Visualise collocation distribution
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(XY_c[:, 0], XY_c[:, 1], marker='o', alpha=0.1, color='blue',  s=1)
    ax.scatter(WALL[:, 0], WALL[:, 1], marker='o', alpha=0.2, color='green', s=2)
    ax.scatter(OUTLET[:, 0], OUTLET[:, 1], marker='o', alpha=0.2, color='orange', s=2)
    ax.scatter(INLET[:, 0],  INLET[:, 1],  marker='o', alpha=0.2, color='red',    s=2)
    plt.show()

    # ---- Build and train model ----
    # Train from scratch:
    model = PINN_laminar_flow(XY_c, INLET, OUTLET, WALL, uv_layers, lb, ub)

    # Load a previously saved model:
    # model = PINN_laminar_flow(XY_c, INLET, OUTLET, WALL, uv_layers, lb, ub,
    #                           ExistModel=1, uvDir='uvNN.pickle')

    start_time = time.time()
    loss_WALL, loss_INLET, loss_OUTLET, loss_f, loss = model.train(
        iter=10000, learning_rate=5e-4)
    model.train_bfgs()
    print("--- %.1f seconds ---" % (time.time() - start_time))

    # Save trained model
    model.save_NN('uvNN.pickle')

    # Persist loss history
    with open('loss_history.pickle', 'wb') as f:
        pickle.dump(model.loss_rec, f)

    # ---- Post-processing ----

    # Load reference (Fluent) solution
    x_FL, y_FL, u_FL, v_FL, p_FL = preprocess(dir='../FluentReferenceMu002/FluentSol.mat')
    field_FLUENT = [x_FL, y_FL, u_FL, v_FL, p_FL]

    # PINN prediction on a uniform grid
    x_pred = np.linspace(0, 1.1, 251)
    y_pred = np.linspace(0, 0.41, 101)
    x_pred, y_pred = np.meshgrid(x_pred, y_pred)
    x_pred = x_pred.flatten()[:, None]
    y_pred = y_pred.flatten()[:, None]
    dst    = ((x_pred - 0.2) ** 2 + (y_pred - 0.2) ** 2) ** 0.5
    x_pred = x_pred[dst >= 0.05].reshape(-1, 1)
    y_pred = y_pred[dst >= 0.05].reshape(-1, 1)

    u_pred, v_pred, p_pred = model.predict(x_pred, y_pred)
    field_MIXED = [x_pred, y_pred, u_pred, v_pred, p_pred]

    postProcess(xmin=0, xmax=1.1, ymin=0, ymax=0.41,
                field_FLUENT=field_FLUENT, field_MIXED=field_MIXED,
                s=3, alpha=0.5)