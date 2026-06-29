import numpy as np
import time
from pyDOE2 import lhs
import matplotlib
matplotlib.use('Agg')  # headless: no display required
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
        self.loss_rec_baseline = []

        # ── Per-term loss histories ──────────────────────────────────────
        # These accumulate across BOTH the Adam phase and the L-BFGS-B
        # phase, giving a complete training trajectory of every loss
        # component. Saved together with self.loss_rec in the pickle.
        self.loss_f_hist      = []
        self.loss_WALL_hist   = []
        self.loss_INLET_hist  = []
        self.loss_OUTLET_hist = []


        # Initialise (or load) weights and biases
        if ExistModel == 0:
            self.uv_weights, self.uv_biases = self.initialize_NN(uv_layers)
        else:
            print("Loading uv NN ...")
            self.uv_weights, self.uv_biases = self.load_NN(uvDir, uv_layers)

        # Collect all trainable variables in one flat list for the optimisers
        self.trainable_variables = self.uv_weights + self.uv_biases

        # ── CACHED boundary tensors ──────────────────────────────────────
        # The boundary point arrays (wall, inlet, outlet) NEVER change
        # during training, so we allocate them on the GPU exactly ONCE
        # here.  Doing this avoids creating fresh tf.Variable objects
        # every iteration of compute_loss(), which was the main cause of
        # the GPU OOM seen with long training runs / large clouds.
        self._x_W_var = tf.Variable(self.x_WALL,   trainable=False, name='x_W')
        self._y_W_var = tf.Variable(self.y_WALL,   trainable=False, name='y_W')
        self._x_I_var = tf.Variable(self.x_INLET,  trainable=False, name='x_I')
        self._y_I_var = tf.Variable(self.y_INLET,  trainable=False, name='y_I')
        self._x_O_var = tf.Variable(self.x_OUTLET, trainable=False, name='x_O')
        self._y_O_var = tf.Variable(self.y_OUTLET, trainable=False, name='y_O')
        self._u_I_const = tf.constant(self.u_INLET, dtype=tf.float32)
        self._v_I_const = tf.constant(self.v_INLET, dtype=tf.float32)

        # ── REUSABLE collocation tensor (mutable, resized by .assign) ────
        # We allocate a tf.Variable for x_c / y_c here and update its
        # contents via .assign() whenever the cloud changes.  This pattern
        # reuses GPU memory instead of repeatedly allocating new buffers.
        # validate_shape=False lets us assign arrays of different sizes
        # (needed because adaptive refinement grows/shrinks the cloud).
        self._x_c_var = tf.Variable(self.x_c, trainable=False,
                                    shape=tf.TensorShape([None, 1]),
                                    name='x_c')
        self._y_c_var = tf.Variable(self.y_c, trainable=False,
                                    shape=tf.TensorShape([None, 1]),
                                    name='y_c')

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

    def dump_history(self, path='loss_history.pickle'):
        """Checkpoint the loss history to disk so the loss-vs-time curve is
        recoverable mid-run (pods get killed at the 6h Nautilus deadline)."""
        with open(path, 'wb') as f:
            pickle.dump({
                'loss':        list(self.loss_rec),
                'loss_f':      list(self.loss_f_hist),
                'loss_WALL':   list(self.loss_WALL_hist),
                'loss_INLET':  list(self.loss_INLET_hist),
                'loss_OUTLET': list(self.loss_OUTLET_hist),
                'n_adam':      getattr(self, 'n_adam', None),
            }, f)
        # Also checkpoint weights so L2 error stays recomputable if the pod
        # is killed mid-run.
        try:
            with open('uvNN_ckpt.pickle', 'wb') as f:
                pickle.dump([[w.numpy() for w in self.uv_weights],
                             [b.numpy() for b in self.uv_biases]], f)
        except Exception:
            pass

    def load_NN(self, fileDir, layers):
        weights, biases = [], []
        with open(fileDir, 'rb') as f:
            uv_w, uv_b = pickle.load(f)

        # Infer the actual architecture from the saved weights rather than
        # asserting against the passed-in layers list.  This avoids
        # AssertionError when the pickle was saved with a different config.
        inferred = [uv_w[0].shape[0]] + [w.shape[1] for w in uv_w]
        if inferred != list(layers):
            print(f"WARNING: uv_layers in script {list(layers)} does not match "
                  f"pickle architecture {inferred}.")
            print(f"  => Using architecture from pickle: {inferred}")
            self.uv_layers = inferred   # update so rest of class is consistent

        for i, (w_arr, b_arr) in enumerate(zip(uv_w, uv_b)):
            weights.append(tf.Variable(w_arr, dtype=tf.float32))
            biases.append(tf.Variable(b_arr, dtype=tf.float32))
            print(f" - Layer {i}: W{w_arr.shape} b{b_arr.shape} loaded OK")
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
        # Match input dtype to weight dtype — needed because train_bfgs
        # promotes the network to float64 mid-run for L-BFGS-B.
        target_dtype = self.uv_weights[0].dtype
        if X.dtype != target_dtype:
            X = tf.cast(X, target_dtype)
        lb = tf.cast(self.lb, target_dtype)
        ub = tf.cast(self.ub, target_dtype)

        # TF1 steady version had normalisation commented out; reproduce that:
        #H = X  # no normalisation (matching original SteadyFlowCylinder_mixed.py)
        # To enable normalisation (recommended), replace the line above with:
        H = 2.0 * (X - lb) / (ub - lb) - 1.0

        for W, b in zip(self.uv_weights[:-1], self.uv_biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        Y = tf.matmul(H, self.uv_weights[-1]) + self.uv_biases[-1]
        return Y

    def net_uv(self, x, y):
        """
        Returns u, v, p, s11, s22, s12.
        Stream-function enforces div(v)=0 automatically.
        x and y must be tf.Variable (not tf.constant) for tape.watch to work.
        """
        with tf.GradientTape(persistent=True) as tape:
            tape.watch([x, y])
            xy  = tf.concat([x, y], axis=1)
            out = self.neural_net(xy)
            psi = out[:, 0:1]
            p   = out[:, 1:2]
            s11 = out[:, 2:3]
            s22 = out[:, 3:4]
            s12 = out[:, 4:5]
        u =  tape.gradient(psi, y)
        v = -tape.gradient(psi, x)
        del tape
        return u, v, p, s11, s22, s12

    def net_f(self, x, y):
        """
        PDE residuals — single-tape implementation.

        Memory-efficient: only ONE forward pass through the network
        (previous version did two passes which doubled GPU usage).

        How a single persistent tape captures everything we need:
        ---------------------------------------------------------
        - tape.watch([x, y]) makes the tape track those inputs.
        - We do ONE forward pass inside the tape.
        - We compute u and v inside the tape using tape.gradient(psi, ...)
          — because the .gradient() call happens while the tape is still
          open (within the `with` block), the resulting u and v tensors
          are themselves recorded as new graph nodes by the same tape.
        - After the tape closes, all derivatives are recoverable.

        This avoids creating a second copy of every intermediate
        activation tensor in GPU memory.
        """
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

    def _sync_colloc_to_gpu(self):
        """
        Push current numpy x_c / y_c into the persistent GPU tensors.
        Called automatically by compute_loss(); also called by
        refine_collocation() and store_base_collocation() whenever the
        cloud changes.
        """
        self._x_c_var.assign(self.x_c)
        self._y_c_var.assign(self.y_c)

    def compute_loss(self):
        """
        Build the total loss using CACHED GPU tensors instead of
        allocating new tf.Variable objects every call.  Drops the
        per-iteration GPU memory footprint by ~6× compared to the
        previous implementation.
        """
        dt   = self.uv_weights[0].dtype          # float32 (Adam) or float64 (L-BFGS)
        x_c  = tf.constant(self.x_c, dtype=dt)
        y_c  = tf.constant(self.y_c, dtype=dt)
        x_W  = tf.constant(self.x_WALL, dtype=dt);  y_W = tf.constant(self.y_WALL, dtype=dt)
        x_I  = tf.constant(self.x_INLET, dtype=dt); y_I = tf.constant(self.y_INLET, dtype=dt)
        u_I  = tf.constant(self.u_INLET, dtype=dt); v_I = tf.constant(self.v_INLET, dtype=dt)
        x_O  = tf.constant(self.x_OUTLET, dtype=dt);y_O = tf.constant(self.y_OUTLET, dtype=dt)

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

    def train(self, iter, learning_rate,
              refine_every=None, refine_kwargs=None, baseline_loss_every=None):
        """
        Adam training loop with optional neighborhood-based residual refinement.

        How the refinement works
        ------------------------
        - The original collocation cloud is stored via store_base_collocation()
          so it can be restored later via reset_collocation_to_base(); however
          base interior points are NOT protected from pruning during training.
        - Every `refine_every` iterations, refine_collocation() is called:
            * Seeds = top fraction of points by residual
            * New candidates are drawn from a small disk around each seed
            * Candidates with residual above `accept_quantile` are ADDED
            * Existing points with residual below `prune_quantile` are
              DELETED, including base interior points if their residual is
              low.  Only the boundary anchors (wall / inlet / outlet) are
              protected because they enforce the BCs through the loss.
        - Result: the cloud reorganises freely around the active flow
          features rather than being held back by the initial sampling.

        Parameters
        ----------
        iter            : number of Adam iterations
        learning_rate   : Adam learning rate
        refine_every    : refine every N iterations.
                          Set to None to disable refinement entirely.
        refine_kwargs   : dict passed to refine_collocation().
                          Must include: lb, ub.
                          Optional: n_seed_neighbors, neighbor_radius,
                                    top_seed_frac, accept_quantile,
                                    prune_quantile, base_protect_frac,
                                    n_exploration_pts, xc, yc, r_cyl,
                                    max_cloud_size, residual_chunk.
        """
        optimizer = tf.keras.optimizers.Adam(learning_rate=learning_rate)

        @tf.function
        def train_step():
            with tf.GradientTape() as tape:
                loss, loss_f, loss_W, loss_I, loss_O = self.compute_loss()
            grads = tape.gradient(loss, self.trainable_variables)
            optimizer.apply_gradients(zip(grads, self.trainable_variables))
            return loss, loss_f, loss_W, loss_I, loss_O

        # We periodically call gc.collect() to free numpy arrays that
        # accumulate when refine_collocation runs.  Python's gc is fine on
        # its own but explicit collection at scheduled points trims peak
        # memory between refinements.
        import gc

        for it in range(iter):

            # ── Gradient step (uses cached GPU tensors) ────────────────
            loss, loss_f, loss_W, loss_I, loss_O = train_step()

            if it % 10 == 0:
                print(f'It: {it}, Loss: {loss.numpy():.3e}  '
                      f'(f={loss_f.numpy():.2e} '
                      f'W={loss_W.numpy():.2e} '
                      f'I={loss_I.numpy():.2e} '
                      f'O={loss_O.numpy():.2e})')

            # Persist EVERY loss component to instance lists so the L-BFGS-B
            # phase can extend them and they all survive to the final pickle.
            self.loss_rec.append(float(loss.numpy()))
            self.loss_f_hist.append(float(loss_f.numpy()))
            self.loss_WALL_hist.append(float(loss_W.numpy()))
            self.loss_INLET_hist.append(float(loss_I.numpy()))
            self.loss_OUTLET_hist.append(float(loss_O.numpy()))

            # Light periodic cleanup — every 200 steps
            if it > 0 and it % 200 == 0:
                gc.collect()

            if it > 0 and it % 1000 == 0:
                self.dump_history()          # periodic checkpoint

        self.n_adam = len(self.loss_rec)     # marks Adam→L-BFGS boundary
        self.dump_history()
        return (self.loss_WALL_hist, self.loss_INLET_hist, self.loss_OUTLET_hist,
                self.loss_f_hist, self.loss_rec)

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
            loss, loss_f, loss_W, loss_I, loss_O = self.compute_loss()
        grads = tape.gradient(loss, self.trainable_variables)
        grad_flat = np.concatenate(
            [g.numpy().flatten() for g in grads])
        self.count += 1
        if self.count % 100 == 0:
            print(f'{self.count} th iterations (L-BFGS), Loss: {loss.numpy():.6e}')
        # Continue the per-term histories so the pickle covers Adam + L-BFGS-B
        self.loss_rec.append(float(loss.numpy()))
        self.loss_f_hist.append(float(loss_f.numpy()))
        self.loss_WALL_hist.append(float(loss_W.numpy()))
        self.loss_INLET_hist.append(float(loss_I.numpy()))
        self.loss_OUTLET_hist.append(float(loss_O.numpy()))
        return loss.numpy().astype(np.float64), grad_flat.astype(np.float64)

    def _set_precision(self, dtype):
        """
        Recreate all trainable variables (and bound constants) in `dtype`.

        L-BFGS-B (scipy) optimises in float64 and expects a smooth float64
        objective.  If the network stays float32, the loss it sees has a
        ~1e-6 noise floor (float32 epsilon), the line search stalls after a
        few iterations, and the loss freezes around 1e-2.  Promoting the
        whole network to float64 removes that noise floor and lets L-BFGS-B
        drive the loss down to 1e-4 and below.
        """
        self.uv_weights = [tf.Variable(tf.cast(w, dtype)) for w in self.uv_weights]
        self.uv_biases  = [tf.Variable(tf.cast(b, dtype)) for b in self.uv_biases]
        self.trainable_variables = self.uv_weights + self.uv_biases

    def train_bfgs(self):
        from scipy.optimize import minimize

        # ── Promote network to float64 so L-BFGS-B can actually converge ──
        self._set_precision(tf.float64)

        self.iter_count = 0

        def callback(xk):
            # Fired once per real L-BFGS-B iteration (not per function eval).
            # loss_rec[-1] is the most recent loss from _loss_and_grad.
            self.iter_count += 1
            if self.iter_count % 100 == 0:
                print(f'[L-BFGS] iter {self.iter_count:5d}  '
                  f'loss = {self.loss_rec[-1]:.6e}  (nfev {self.count})')
            if self.iter_count % 200 == 0:
                self.dump_history()          # periodic checkpoint

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
                'maxiter': int(os.environ.get('SWEEP_LBFGS_MAXITER', 100000)),
                'maxfun':  int(os.environ.get('SWEEP_LBFGS_MAXITER', 100000)),
                'maxcor':  50,
                'maxls':   50,
                'ftol':    1e-10,
                'gtol':    1e-7
                # 'ftol':    1.0 * np.finfo(float).eps,
                # 'gtol':    1e-8,
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
        np_dtype = self.uv_weights[0].dtype.as_numpy_dtype
        x = tf.constant(np.asarray(x_star, dtype=np_dtype))
        y = tf.constant(np.asarray(y_star, dtype=np_dtype))
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

def relative_l2_error(model, ref_mat_path='FluentReferenceMu002/FluentSol.mat',
                      r_cyl=0.05, xc=0.2, yc=0.2, predict_chunk=20_000):
    """
    Compute relative L2 error of PINN prediction vs Fluent reference.

    Implements equation (9) from Rao, Sun & Liu (2020):

        eps = sqrt(sum |f_pred - f_ref|^2)  /  sqrt(sum |f_ref|^2)

    Computed independently for u, v, p plus a combined velocity-magnitude
    error.  Errors are evaluated AT the Fluent mesh nodes — we query the
    PINN at every Fluent (x, y) and compare against the Fluent value
    stored there, so there is no interpolation error introduced.

    Parameters
    ----------
    model           : trained PINN_laminar_flow instance with .predict()
    ref_mat_path    : path to the Fluent reference .mat file
                      (expects keys x, y, vx, vy, p)
    r_cyl, xc, yc   : cylinder radius and centre — Fluent nodes that fall
                      inside the cylinder (if any) are masked out before
                      computing the error.
    pressure_shift  : if True, subtract the mean from both p_pred and
                      p_ref before computing the pressure error.  This
                      removes the arbitrary additive constant inherent
                      to incompressible-flow pressure fields (since p is
                      determined only up to a constant when the only BC
                      is p=0 at the outlet, and Fluent / PINN may pick
                      different references).
    predict_chunk   : chunk size for PINN predictions — guards against
                      GPU OOM when the Fluent mesh has many nodes.

    Returns
    -------
    errors : dict with keys 'u', 'v', 'p', 'speed' mapping to the
             relative L2 error of each quantity (Python floats).
             Also prints a summary table to stdout.
    """
    # ── Load Fluent reference ────────────────────────────────────────────
    x_ref, y_ref, vx_ref, vy_ref, p_ref = preprocess(ref_mat_path)

    # ── Exclude any reference nodes inside the cylinder (sanity guard) ──
    dist = np.sqrt((x_ref - xc) ** 2 + (y_ref - yc) ** 2)
    keep = dist.flatten() >= r_cyl
    x_ref  = x_ref[keep].reshape(-1, 1)
    y_ref  = y_ref[keep].reshape(-1, 1)
    vx_ref = vx_ref[keep].reshape(-1, 1)
    vy_ref = vy_ref[keep].reshape(-1, 1)
    p_ref  = p_ref[keep].reshape(-1, 1)
    n_pts  = len(x_ref)

    # ── PINN prediction at Fluent nodes (chunked to limit GPU memory) ───
    u_pred = np.empty_like(vx_ref)
    v_pred = np.empty_like(vy_ref)
    p_pred = np.empty_like(p_ref)
    for s in range(0, n_pts, predict_chunk):
        e = min(s + predict_chunk, n_pts)
        u_pred[s:e], v_pred[s:e], p_pred[s:e] = model.predict(
            x_ref[s:e], y_ref[s:e])

    # ── Relative L2 error per quantity ──────────────────────────────────
    def _rel_l2(pred, ref):
        num = np.sqrt(np.sum((pred - ref) ** 2))
        den = np.sqrt(np.sum(ref ** 2))
        return float(num / (den + 1e-30))

    err_u     = _rel_l2(u_pred, vx_ref)
    err_v     = _rel_l2(v_pred, vy_ref)
    err_p     = _rel_l2(p_pred, p_ref)

    # ── Console summary ─────────────────────────────────────────────────
    print()
    print("=" * 60)
    print(f"  Relative L2 error vs Fluent reference ({ref_mat_path})")
    print(f"  evaluated at {n_pts} reference nodes outside the cylinder")
    print("=" * 60)
    print(f"    u      :  {err_u:.4e}   ({100*err_u:.2f} %)")
    print(f"    v      :  {err_v:.4e}   ({100*err_v:.2f} %)")
    print(f"    p      :  {err_p:.4e}   ({100*err_p:.2f} %)")
    print("=" * 60)
    print()

    return {'u': err_u, 'v': err_v, 'p': err_p}

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
    XY_c        = lb + (ub - lb) * lhs(2, 20000)
    XY_c        = np.concatenate((XY_c, WALL, CYLD, OUTLET, INLET[:, 0:2]), axis=0)
    XY_c_refine = np.array([0.1, 0.1]) + np.array([0.2, 0.2]) * lhs(2, 10000)
    XY_c        = np.concatenate((XY_c, XY_c_refine), axis=0)
    XY_c        = DelCylPT(XY_c, xc=0.2, yc=0.2, r=0.05)

    print("Total collocation points:", XY_c.shape)

    # Visualise collocation distribution
    fig, ax = plt.subplots()
    ax.set_aspect('equal')
    ax.scatter(XY_c[:, 0], XY_c[:, 1], marker='o', alpha=0.1, color='blue',  s=1)
    ax.scatter(WALL[:, 0], WALL[:, 1], marker='o', alpha=0.2, color='green', s=2)
    ax.scatter(OUTLET[:, 0], OUTLET[:, 1], marker='o', alpha=0.2, color='orange', s=2)
    ax.scatter(INLET[:, 0],  INLET[:, 1],  marker='o', alpha=0.2, color='red',    s=2)
    plt.tight_layout()
    plt.savefig('colloc_points.png', dpi=150)
    plt.close()
    print('Saved colloc_points.png')

    # ---- Build and train model ----
    # Train from scratch:
    model = PINN_laminar_flow(XY_c, INLET, OUTLET, WALL, uv_layers, lb, ub)

    start_time = time.time()
    loss_WALL, loss_INLET, loss_OUTLET, loss_f, loss = model.train(
        iter=int(os.environ.get('SWEEP_ITERS', 10000)),
        learning_rate=1e-3,
    )

    # ── Visualise final adaptive collocation distribution ────────────────
    fig, ax = plt.subplots(figsize=(9, 3))
    ax.set_aspect('equal')
    ax.scatter(model.x_c, model.y_c, s=0.5, alpha=0.2,
               color='steelblue', label='colloc (base + refined)')
    ax.scatter(WALL[:, 0],   WALL[:, 1],   s=2, color='green',  alpha=0.5, label='wall')
    ax.scatter(OUTLET[:, 0], OUTLET[:, 1], s=2, color='orange', alpha=0.5, label='outlet')
    ax.scatter(INLET[:, 0],  INLET[:, 1],  s=2, color='red',    alpha=0.5, label='inlet')
    ax.legend(loc='upper right', markerscale=4, fontsize=7)
    ax.set_title('Collocation points after additive refinement')
    plt.tight_layout()
    plt.savefig('colloc_adaptive.png', dpi=150)
    plt.close()
    print('Saved colloc_adaptive.png')

    # ── L-BFGS-B fine convergence on the final point set ─────────────────
    model.train_bfgs()
    print("--- %.1f seconds ---" % (time.time() - start_time))

    # Save trained model
    model.save_NN('uvNN.pickle')

    # Persist loss history — saved as a dict containing ALL components
    # so the consumer side can do  d = pickle.load(...)  and read e.g.
    # d['loss'], d['loss_f'], d['loss_WALL'], d['loss_INLET'], d['loss_OUTLET'].
    # All five lists cover both the Adam and L-BFGS-B phases.
    loss_history = {
        'loss':        list(model.loss_rec),
        'loss_f':      list(model.loss_f_hist),
        'loss_WALL':   list(model.loss_WALL_hist),
        'loss_INLET':  list(model.loss_INLET_hist),
        'loss_OUTLET': list(model.loss_OUTLET_hist),
    }
    with open('loss_history.pickle', 'wb') as f:
        pickle.dump(loss_history, f)
    print(f"Saved loss_history.pickle  (keys: {list(loss_history.keys())}, "
          f"length: {len(loss_history['loss'])})")
    
    errors = relative_l2_error(model, ref_mat_path='FluentReferenceMu002/FluentSol.mat',
                               r_cyl=0.05, xc=0.2, yc=0.2,)

    # ---- Post-processing ----

    # Load reference (Fluent) solution
    x_FL, y_FL, u_FL, v_FL, p_FL = preprocess(dir='FluentReferenceMu002/FluentSol.mat')
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