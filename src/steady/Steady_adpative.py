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
        # Accumulate across BOTH the Adam phase and the L-BFGS-B phase so
        # the final pickle contains every loss component for the whole run.
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
        # is killed mid-run (we lost the adaptive weights once this way).
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
        # Match input dtype to weight dtype — needed because train_bfgs
        # promotes weights to float64 mid-run.
        target_dtype = self.uv_weights[0].dtype
        if X.dtype != target_dtype:
            X = tf.cast(X, target_dtype)
        lb = tf.cast(self.lb, target_dtype)
        ub = tf.cast(self.ub, target_dtype)

        H = 2.0 * (X - lb) / (ub - lb) - 1.0

        for W, b in zip(self.uv_weights[:-1], self.uv_biases[:-1]):
            H = tf.tanh(tf.matmul(H, W) + b)
        Y = tf.matmul(H, self.uv_weights[-1]) + self.uv_biases[-1]
        return Y

    def net_uv(self, x, y):
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
    # Residual-based adaptive collocation (RAR — Residual Adaptive Refinement)
    # ------------------------------------------------------------------

    def store_base_collocation(self, XY_c_base, XY_c_org):
        self._XY_c_base = XY_c_base.astype(np.float32).copy()
        self._XY_c_org = XY_c_org.astype(np.float32).copy()
        print(f"  [adaptive] base collocation set stored: "
              f"{len(self._XY_c_org)} pts (will never be removed)")

    def _residual_chunked(self, xy, chunk=10000):
        np_dtype = self.uv_weights[0].dtype.as_numpy_dtype
        xy = np.asarray(xy, dtype=np_dtype)
        N  = len(xy)
        out = np.empty(N, dtype=np_dtype)
        for s in range(0, N, chunk):
            e = min(s + chunk, N)
            x_t = tf.convert_to_tensor(xy[s:e, 0:1])
            y_t = tf.convert_to_tensor(xy[s:e, 1:2])
            f_u, f_v, f_s11, f_s22, f_s12, f_p = self.net_f(x_t, y_t)
            r = (tf.square(f_u)   + tf.square(f_v)
               + tf.square(f_s11) + tf.square(f_s22)
               + tf.square(f_s12) + tf.square(f_p))
            out[s:e] = r.numpy().flatten()
            del f_u, f_v, f_s11, f_s22, f_s12, f_p, r, x_t, y_t
        return out

    # ── Internal helpers for residual refinement ──────────────────────────

    def _build_protected_xy(self, base_protect_frac):
        if hasattr(self, '_protected_xy'):
            return self._protected_xy

        anchors = [
            np.column_stack([self.x_WALL.flatten(),   self.y_WALL.flatten()]),
            np.column_stack([self.x_INLET.flatten(),  self.y_INLET.flatten()]),
            np.column_stack([self.x_OUTLET.flatten(), self.y_OUTLET.flatten()]),
        ]
        n_bound = sum(len(a) for a in anchors)

        if base_protect_frac > 0.0 and hasattr(self, '_XY_c_base'):
            n_base    = len(self._XY_c_org)
            n_protect = int(base_protect_frac * n_base)
            idx       = np.random.choice(n_base, size=n_protect, replace=False)
            anchors.append(self._XY_c_org[idx])
            print(f"  [adaptive] protected set locked in: "
                  f"{n_bound} boundary + {n_protect}/{n_base} base coverage")
        else:
            print(f"  [adaptive] protected set locked in: "
                  f"{n_bound} boundary anchors only")

        self._protected_xy = np.concatenate(anchors, axis=0).astype(np.float32)
        return self._protected_xy

    def _protected_mask(self, XY, protected_xy, tol=1e-6):
        try:
            from scipy.spatial import cKDTree
        except ImportError:
            return np.zeros(len(XY), dtype=bool)
        tree = cKDTree(protected_xy)
        d, _ = tree.query(XY, k=1)
        return d < tol

    def refine_collocation(self, lb, ub,
                           n_seed_neighbors=10,
                           neighbor_radius=0.02,
                           top_seed_frac=0.10,
                           accept_quantile=0.70,
                           prune_quantile=0.10,
                           base_protect_frac=0.6,
                           n_exploration_pts=2000,
                           xc=0.2, yc=0.2, r_cyl=0.05,
                           max_cloud_size=120_000,
                           residual_chunk=10_000):

        # ── 1. Residual on current cloud ─────────────────────────────────
        XY_now = np.concatenate([self.x_c, self.y_c], axis=1).astype(np.float32)
        res    = self._residual_chunked(XY_now, chunk=residual_chunk)

        thr_accept = np.quantile(res, accept_quantile)
        thr_prune  = np.quantile(res, prune_quantile)

        # ── 2. Build / look up protected anchors, mark cloud points ──────
        protected_xy   = self._build_protected_xy(base_protect_frac)
        protected_mask = self._protected_mask(XY_now, protected_xy)

        # ── 3. Seeds = top fraction by residual ──────────────────────────
        n_seeds  = max(1, int(top_seed_frac * len(XY_now)))
        seed_idx = np.argpartition(res, -n_seeds)[-n_seeds:]
        seeds    = XY_now[seed_idx]

        # ── 4. Local candidates (disks around seeds) ─────────────────────
        n_total = n_seeds * n_seed_neighbors
        rng_r   = neighbor_radius * np.sqrt(np.random.rand(n_total))
        rng_th  = 2 * np.pi * np.random.rand(n_total)
        seeds_rep  = np.repeat(seeds, n_seed_neighbors, axis=0)
        local_cand = seeds_rep + np.column_stack([rng_r * np.cos(rng_th),
                                                  rng_r * np.sin(rng_th)])

        # ── 5. Exploration candidates (global LHS) ───────────────────────
        if n_exploration_pts > 0:
            explore = lb + (ub - lb) * lhs(2, n_exploration_pts)
            cand = np.concatenate([local_cand, explore.astype(np.float32)],
                                  axis=0)
        else:
            cand = local_cand

        # Clip to domain, reject cylinder interior
        cand[:, 0] = np.clip(cand[:, 0], lb[0], ub[0])
        cand[:, 1] = np.clip(cand[:, 1], lb[1], ub[1])
        dist_cyl = np.sqrt((cand[:, 0] - xc)**2 + (cand[:, 1] - yc)**2)
        cand = cand[dist_cyl > r_cyl].astype(np.float32)
        if len(cand) == 0:
            print("  [adaptive] no valid candidates this round")
            return XY_now

        # ── 6. Residual on candidates; accept high-residual ones ────────
        cand_res = self._residual_chunked(cand, chunk=residual_chunk)
        accept   = cand_res >= thr_accept
        new_pts  = cand[accept]

        # ── 7. Prune cold spots (excluding protected anchors) ────────────
        prune_mask = (res < thr_prune) & (~protected_mask)
        XY_kept    = XY_now[~prune_mask]

        # ── 8. Combine kept + new points; deduplicate ───────────────────
        if len(new_pts) > 0:
            XY_new = np.concatenate([XY_kept, new_pts], axis=0).astype(np.float32)
        else:
            XY_new = XY_kept.astype(np.float32)

        _, uniq_idx = np.unique(np.round(XY_new, 5), axis=0, return_index=True)
        XY_new = XY_new[np.sort(uniq_idx)]

        # ── 9. Enforce max_cloud_size cap if exceeded ───────────────────
        if len(XY_new) > max_cloud_size:
            res_new  = self._residual_chunked(XY_new, chunk=residual_chunk)
            prot_new = self._protected_mask(XY_new, protected_xy)
            # Keep protected first, then highest residual
            order    = np.lexsort((-res_new, ~prot_new))
            keep_idx = order[:max_cloud_size]
            XY_new   = XY_new[np.sort(keep_idx)]
            print(f"  [adaptive] cap enforced: trimmed to {len(XY_new)}")

        # ── 10. Commit ───────────────────────────────────────────────────
        self.x_c = XY_new[:, 0:1]
        self.y_c = XY_new[:, 1:2]
        self._sync_colloc_to_gpu()

        print(f"  [adaptive] {len(XY_now)} -> {len(XY_new)}  "
              f"(+{int(accept.sum())} added, "
              f"-{int(prune_mask.sum())} pruned, "
              f"protected={int(protected_mask.sum())}, "
              f"thr_accept={thr_accept:.2e})")
        return XY_new

    def reset_collocation_to_base(self):
        if not hasattr(self, '_XY_c_base'):
            print("  [adaptive] no base set stored — nothing to reset")
            return
        self.x_c = self._XY_c_org[:, 0:1].copy()
        self.y_c = self._XY_c_org[:, 1:2].copy()
        print(f"  [adaptive] reset to base set ({len(self._XY_c_org)} pts)")

    # ------------------------------------------------------------------
    # Loss function
    # ------------------------------------------------------------------

    def _sync_colloc_to_gpu(self):
        np_dtype = self._x_c_var.dtype.as_numpy_dtype
        self._x_c_var.assign(self.x_c.astype(np_dtype))
        self._y_c_var.assign(self.y_c.astype(np_dtype))

    def compute_loss(self):
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

        loss = loss_f + loss_WALL + loss_INLET + loss_OUTLET
        return loss, loss_f, loss_WALL, loss_INLET, loss_OUTLET

    def compute_loss_baseline(self):
        """
        Same loss formula but evaluated on the ORIGINAL (frozen) base
        collocation cloud, for tracking how loss on the fixed-points
        baseline evolves during adaptive training.  Allocated tensors
        are stored once on the model the first time this is called.
        """
        # Lazy one-time allocation of the baseline tensors
        if not hasattr(self, '_x_c_base_var'):
            self._x_c_base_var = tf.Variable(
                self._XY_c_base[:, 0:1], trainable=False, name='x_c_base')
            self._y_c_base_var = tf.Variable(
                self._XY_c_base[:, 1:2], trainable=False, name='y_c_base')

        x_c, y_c = self._x_c_base_var, self._y_c_base_var
        x_W, y_W = self._x_W_var, self._y_W_var
        x_I, y_I = self._x_I_var, self._y_I_var
        x_O, y_O = self._x_O_var, self._y_O_var
        u_I, v_I = self._u_I_const, self._v_I_const

        f_u, f_v, f_s11, f_s22, f_s12, f_p = self.net_f(x_c, y_c)
        loss_f = (tf.reduce_mean(tf.square(f_u))
                + tf.reduce_mean(tf.square(f_v))
                + tf.reduce_mean(tf.square(f_s11))
                + tf.reduce_mean(tf.square(f_s22))
                + tf.reduce_mean(tf.square(f_s12))
                + tf.reduce_mean(tf.square(f_p)))

        u_W, v_W, _, _, _, _ = self.net_uv(x_W, y_W)
        loss_WALL = (tf.reduce_mean(tf.square(u_W))
                   + tf.reduce_mean(tf.square(v_W)))

        u_Ip, v_Ip, _, _, _, _ = self.net_uv(x_I, y_I)
        loss_INLET = (tf.reduce_mean(tf.square(u_Ip - u_I))
                    + tf.reduce_mean(tf.square(v_Ip - v_I)))

        _, _, p_O, _, _, _ = self.net_uv(x_O, y_O)
        loss_OUTLET = tf.reduce_mean(tf.square(p_O))

        loss = loss_f + loss_WALL + loss_INLET + loss_OUTLET
        return loss, loss_f, loss_WALL, loss_INLET, loss_OUTLET

    # ------------------------------------------------------------------
    # Adam training
    # ------------------------------------------------------------------

    def train(self, iter, learning_rate, start_refine_iter=None,
              refine_every=None, refine_kwargs=None, baseline_loss_every=None):
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

        # Track baseline loss history if requested
        baseline_hist = []

        for it in range(iter):
            if (refine_every is not None
                    and it > start_refine_iter
                    and it % refine_every == 0
                    and refine_kwargs is not None):
                print(f'  [iter {it}] refining (adding high-residual pts) ...')
                self.refine_collocation(**refine_kwargs)
                gc.collect()   # release numpy intermediates from refinement

            if (baseline_loss_every is not None
                    and baseline_loss_every > 0
                    and it > start_refine_iter
                    and it % baseline_loss_every == 0
                    and hasattr(self, '_XY_c_base')):
                bl_loss, bl_f, bl_W, bl_I, bl_O = self.compute_loss_baseline()
                baseline_hist.append((it, float(bl_loss.numpy()),
                                          float(bl_f.numpy()),
                                          float(bl_W.numpy()),
                                          float(bl_I.numpy()),
                                          float(bl_O.numpy())))
                print("\033[1m-------------Baseline Loss---------------\033[0m\n"
                      f'It: {it}, Loss: {bl_loss.numpy():.3e}  '
                      f'(f={bl_f.numpy():.2e} '
                      f'W={bl_W.numpy():.2e} '
                      f'I={bl_I.numpy():.2e} '
                      f'O={bl_O.numpy():.2e})\n'
                      "\033[1m-------------Baseline Loss---------------\033[0m")


            loss, loss_f, loss_W, loss_I, loss_O = train_step()

            if it % 10 == 0:
                print(f'It: {it}, Loss: {loss.numpy():.3e}  '
                      f'(f={loss_f.numpy():.2e} '
                      f'W={loss_W.numpy():.2e} '
                      f'I={loss_I.numpy():.2e} '
                      f'O={loss_O.numpy():.2e})')

            self.loss_rec.append(float(loss.numpy()))
            self.loss_f_hist.append(float(loss_f.numpy()))
            self.loss_WALL_hist.append(float(loss_W.numpy()))
            self.loss_INLET_hist.append(float(loss_I.numpy()))
            self.loss_OUTLET_hist.append(float(loss_O.numpy()))

            if it > 0 and it % 200 == 0:
                gc.collect()

            if it > 0 and it % 1000 == 0:
                self.dump_history()          # periodic checkpoint

        self.n_adam = len(self.loss_rec)     # marks Adam→L-BFGS boundary
        self.dump_history()
        self.baseline_hist = baseline_hist
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
            loss, _, _, _, _ = self.compute_loss()
        grads = tape.gradient(loss, self.trainable_variables)
        grad_flat = np.concatenate(
            [g.numpy().flatten() for g in grads])
        self.count += 1
        if self.count % 100 == 0:
            print(f'{self.count} th iterations (L-BFGS), Loss: {loss.numpy():.6e}')
        self.loss_rec.append(loss.numpy())
        return loss.numpy().astype(np.float64), grad_flat.astype(np.float64)

    def _set_precision(self, dtype):
        """
        Recreate all trainable variables in `dtype`.

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
            self.iter_count += 1
            if self.iter_count % 100 == 0:
                print(f'  [L-BFGS] iter {self.iter_count:5d}  '
                      f'loss = {self.loss_rec[-1]:.6e}  '
                      f'(nfev {self.count})')
            if self.iter_count % 200 == 0:
                self.dump_history()          # periodic checkpoint

        print('\n>>> Starting L-BFGS-B optimisation (float64)')
        print(f'    Loss before L-BFGS: {self.loss_rec[-1]:.6e}')

        x0 = self._pack_variables().astype(np.float64)
        print(f'    parameters: {len(x0)}')

        try:
            result = minimize(
                self._loss_and_grad,
                x0,
                method='L-BFGS-B',
                jac=True,
                callback=callback,
                options={
                    'maxiter': int(os.environ.get('SWEEP_LBFGS_MAXITER', 50000)),
                    'maxfun':  int(os.environ.get('SWEEP_LBFGS_MAXITER', 50000)),
                    'maxcor':  50,
                    'maxls':   50,
                    'ftol':    1e-10,
                    'gtol':    1e-7,
                    'disp':    True,
                },
            )
        except Exception as e:
            print(f'  [L-BFGS-B] FAILED: {type(e).__name__}: {e}')
            return

        # Restore the BEST point scipy found and append one final loss eval
        self._unpack_variables(result.x)
        loss, loss_f, loss_W, loss_I, loss_O = self.compute_loss()
        self.loss_rec.append(float(loss.numpy()))
        self.loss_f_hist.append(float(loss_f.numpy()))
        self.loss_WALL_hist.append(float(loss_W.numpy()))
        self.loss_INLET_hist.append(float(loss_I.numpy()))
        self.loss_OUTLET_hist.append(float(loss_O.numpy()))

        print(f'    L-BFGS-B finished: {result.message}')
        print(f'    Loss after  L-BFGS: {float(loss.numpy()):.6e}')
        print(f'    Total iters: {self.iter_count}, total nfev: {self.count}')

    # ------------------------------------------------------------------
    # Inference
    # ------------------------------------------------------------------

    def predict(self, x_star, y_star):
        """Inference — auto-matches the current network dtype (float32/64)."""
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
    XY_c = lb + (ub - lb) * lhs(2, 20000)
    XY_c = np.concatenate((XY_c, WALL, CYLD, OUTLET, INLET[:, 0:2]), axis=0)
    XY_c = DelCylPT(XY_c, xc=0.2, yc=0.2, r=0.05)
    
    XY_c_refine = np.array([0.1, 0.1]) + np.array([0.2, 0.2]) * lhs(2, 10000)
    XY_c_baseline = np.concatenate((XY_c, XY_c_refine), axis=0)
    XY_c_baseline = DelCylPT(XY_c_baseline, xc=0.2, yc=0.2, r=0.05)

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

    model.store_base_collocation(XY_c_baseline, XY_c)

    # Sweep overrides: read from env vars, default to the baseline values so
    # running the script directly (no env set) reproduces the baseline run.
    NSEED    = int(os.environ.get('SWEEP_NSEED', 5))
    ACCEPTQ  = float(os.environ.get('SWEEP_ACCEPTQ', 0.95))
    NEXPLORE = int(os.environ.get('SWEEP_NEXPLORE', 1000))
    print(f">>> SWEEP CONFIG: n_seed_neighbors={NSEED}, "
          f"accept_quantile={ACCEPTQ}, n_exploration_pts={NEXPLORE}")

    refine_kwargs = dict(
        lb                   = lb,
        ub                   = ub,
        n_seed_neighbors     = NSEED,
        neighbor_radius      = 0.02,
        top_seed_frac        = 0.10,
        accept_quantile      = ACCEPTQ,
        prune_quantile       = 0.05,
        base_protect_frac    = 0.20,   # 0 = boundaries only; 0.4 = + 40% base
        n_exploration_pts    = NEXPLORE,  # LHS probes over full domain each round
        xc=0.2, yc=0.2, r_cyl=0.05,
        max_cloud_size       = 50000,
    )

    start_time = time.time()
    loss_WALL, loss_INLET, loss_OUTLET, loss_f, loss = model.train(
        iter=int(os.environ.get('SWEEP_ITERS', 10000)),
        learning_rate=1e-3,
        start_refine_iter=2000,
        refine_every=1000,
        refine_kwargs=refine_kwargs,
        baseline_loss_every = 100
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

    # Baseline loss diagnostic stays in its own file (unchanged)
    with open('loss_history_baseline.pickle', 'wb') as f:
        pickle.dump(model.baseline_hist, f)
    
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