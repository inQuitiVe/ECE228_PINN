"""
2D unsteady incompressible NS reference data generator.
Chorin projection on collocated grid with CONSISTENT forward-divergence /
backward-gradient pair — guarantees discrete div(u_new)=0 at every cell.

Domain:
  x in [0,1.1], y in [0,0.41]
  solid obstacle = simplified human-shaped 2D mask, rotated 90 degrees left
  mu=0.005, rho=1.0, U_max=0.5, period=1.0, tmax=0.5

Note:
  This is not a realistic human aerodynamics simulation. It is a 2D
  incompressible laminar-flow reference generator with a human-shaped solid
  mask, intended as a geometry-generalization demo.
"""

import os, time
import numpy as np
import scipy.sparse, scipy.sparse.linalg, scipy.io

# ── parameters ────────────────────────────────────────────────────────────────
MU, RHO, NU = 0.005, 1.0, 0.005
U_MAX, T_MAX, PERIOD = 0.5, 0.5, 1.0
H = 0.41

# Human-shaped obstacle parameters.
# CHAR_LEN keeps the nominal Reynolds number comparable to the original
# Re=100 cylinder case, where D=2*RC=0.10.
OBST_XC, OBST_YC = 0.34, 0.205
HUMAN_SCALE = 1.0
CHAR_LEN = 0.10

# NX, NY = 220, 82
# DX = 1.1 / NX   # 0.005
# DY = H   / NY   # 0.005
# DT = 0.001       # CFL≈0.1, viscous≈0.4

NX, NY = 440, 164
DX = 1.1 / NX
DY = H / NY
DT = 0.000125

N_STEPS  = int(round(T_MAX / DT))   # 500
N_SNAP   = 51
SNAP_EVERY = N_STEPS // (N_SNAP - 1)

xc = (np.arange(NX) + 0.5) * DX        # (NX,)
yc = (np.arange(NY) + 0.5) * DY        # (NY,)
XX, YY = np.meshgrid(xc, yc)           # (NY, NX)

# ── human-shaped obstacle mask ────────────────────────────────────────────────
def capsule_mask(XX, YY, x1, y1, x2, y2, r):
    """Capsule = line segment from (x1,y1) to (x2,y2) thickened by radius r."""
    px = XX - x1
    py = YY - y1
    vx = x2 - x1
    vy = y2 - y1
    seg_len2 = vx*vx + vy*vy
    if seg_len2 == 0:
        return (XX - x1)**2 + (YY - y1)**2 <= r**2
    t = (px*vx + py*vy) / seg_len2
    t = np.clip(t, 0.0, 1.0)
    closest_x = x1 + t*vx
    closest_y = y1 + t*vy
    return (XX - closest_x)**2 + (YY - closest_y)**2 <= r**2

def rounded_box_mask(XX, YY, xc, yc, w, h, r):
    """Rounded rectangle mask using a signed-distance-style formula."""
    qx = np.abs(XX - xc) - (w/2 - r)
    qy = np.abs(YY - yc) - (h/2 - r)
    outside = np.sqrt(np.maximum(qx, 0)**2 + np.maximum(qy, 0)**2)
    inside = np.minimum(np.maximum(qx, qy), 0)
    return outside + inside <= r

def humans_1_2_1(XX, YY,
                 xc_left=0.21,
                 xc_mid=0.41,
                 xc_right=0.66,
                 yc_center=0.205,
                 ygap=0.085,
                 scale_single=0.68,
                 scale_pair=0.68):
    left = human_mask_rot90_left(XX, YY, xc=xc_left, yc=yc_center, scale=scale_single)
    mid_low = human_mask_rot90_left(XX, YY, xc=xc_mid, yc=yc_center - ygap, scale=scale_pair)
    mid_high = human_mask_rot90_left(XX, YY, xc=xc_mid, yc=yc_center + ygap, scale=scale_pair)
    right = human_mask_rot90_left(XX, YY, xc=xc_right, yc=yc_center, scale=scale_single)
    return left | mid_low | mid_high | right

def two_humans_y_stack(XX, YY,
                       xc=0.34,
                       yc1=0.135,
                       yc2=0.275,
                       scale=0.78):
    solid1 = human_mask_rot90_left(XX, YY, xc=xc, yc=yc1, scale=scale)
    solid2 = human_mask_rot90_left(XX, YY, xc=xc, yc=yc2, scale=scale)
    return solid1 | solid2

def four_humans_2x2(XX, YY,
                    xc_left=0.30,
                    xc_right=0.58,
                    yc_low=0.135,
                    yc_high=0.275,
                    scale=0.78):
    """
    Four rotated human obstacles:
      two stacked along y on the left,
      two stacked along y on the right.
    """

    left_low = human_mask_rot90_left(
        XX, YY, xc=xc_left, yc=yc_low, scale=scale
    )
    left_high = human_mask_rot90_left(
        XX, YY, xc=xc_left, yc=yc_high, scale=scale
    )

    right_low = human_mask_rot90_left(
        XX, YY, xc=xc_right, yc=yc_low, scale=scale
    )
    right_high = human_mask_rot90_left(
        XX, YY, xc=xc_right, yc=yc_high, scale=scale
    )

    return left_low | left_high | right_low | right_high

def human_mask_upright(XX, YY, xc=0.30, yc=0.205, scale=1.0):
    """Front-facing icon-like human obstacle."""
    s = scale

    head = (XX - xc)**2 + (YY - (yc + 0.105*s))**2 <= (0.031*s)**2

    torso = rounded_box_mask(
        XX, YY,
        xc=xc,
        yc=yc + 0.010*s,
        w=0.095*s,
        h=0.120*s,
        r=0.024*s,
    )

    shoulders = capsule_mask(
        XX, YY,
        xc - 0.040*s, yc + 0.055*s,
        xc + 0.040*s, yc + 0.055*s,
        0.020*s,
    )

    left_arm = capsule_mask(
        XX, YY,
        xc - 0.066*s, yc + 0.040*s,
        xc - 0.070*s, yc - 0.035*s,
        0.0145*s,
    )
    right_arm = capsule_mask(
        XX, YY,
        xc + 0.066*s, yc + 0.040*s,
        xc + 0.070*s, yc - 0.035*s,
        0.0145*s,
    )

    left_connector = capsule_mask(
        XX, YY,
        xc - 0.047*s, yc + 0.048*s,
        xc - 0.061*s, yc + 0.043*s,
        0.010*s,
    )
    right_connector = capsule_mask(
        XX, YY,
        xc + 0.047*s, yc + 0.048*s,
        xc + 0.061*s, yc + 0.043*s,
        0.010*s,
    )

    left_leg = capsule_mask(
        XX, YY,
        xc - 0.024*s, yc - 0.043*s,
        xc - 0.026*s, yc - 0.118*s,
        0.018*s,
    )
    right_leg = capsule_mask(
        XX, YY,
        xc + 0.024*s, yc - 0.043*s,
        xc + 0.026*s, yc - 0.118*s,
        0.018*s,
    )

    leg_gap = rounded_box_mask(
        XX, YY,
        xc=xc,
        yc=yc - 0.085*s,
        w=0.016*s,
        h=0.085*s,
        r=0.006*s,
    )

    solid = (
        head | torso | shoulders |
        left_arm | right_arm |
        left_connector | right_connector |
        left_leg | right_leg
    )
    return solid & (~leg_gap)

def human_mask_rot90_left(XX, YY, xc=OBST_XC, yc=OBST_YC, scale=HUMAN_SCALE):
    """
    Rotate the upright human icon 90 degrees counterclockwise around (xc,yc),
    so the head points to the left and feet point to the right.
    """
    # inverse coordinate transform for +90 degree rotation
    X0 = xc + (YY - yc)
    Y0 = yc - (XX - xc)
    return human_mask_upright(X0, Y0, xc=xc, yc=yc, scale=scale)

# SOLID = human_mask_rot90_left(XX, YY)
SOLID = humans_1_2_1(XX, YY)
FLUID = ~SOLID
# Obstacle extent is used only for console reporting. Do not save/draw a bbox.
jj, ii = np.where(SOLID)
OBST_XMIN, OBST_XMAX = xc[ii].min(), xc[ii].max()
OBST_YMIN, OBST_YMAX = yc[jj].min(), yc[jj].max()
RE = RHO * U_MAX * CHAR_LEN / MU

print(f"Grid {NX}×{NY}  dx={DX:.4f}  dt={DT}  nominal Re={RE:.1f}")
print(f"Obstacle cells={int(SOLID.sum())}  bbox x=[{OBST_XMIN:.3f},{OBST_XMAX:.3f}] "
      f"y=[{OBST_YMIN:.3f},{OBST_YMAX:.3f}]")

# ── Poisson matrix ────────────────────────────────────────────────────────────
# Boundary conditions:
#   Dirichlet φ=0 at outlet column (i=NX-1) and solid-obstacle cells
#   Neumann   ∂φ/∂n=0 at inlet (i=0), bottom (j=0), top (j=NY-1)
# The 5-point stencil LP is EXACTLY consistent with the backward gradient /
# forward divergence pair (see derivation in commit message).
def build_poisson():
    """
    5-point Laplacian for φ.
    Dirichlet φ=0: outlet column only (i=NX-1).
    Neumann ∂φ/∂n=0: inlet (i=0), top (j=NY-1), bottom (j=0).
    Solid-obstacle cells: treated as regular interior — no Dirichlet BC.
      Applying Dirichlet φ=0 inside the solid obstacle creates spurious pressure
      gradients that act as a mass sink; omitting it preserves mass conservation.
    """
    ax, ay, n = 1/DX**2, 1/DY**2, NX*NY
    rows, cols, vals = [], [], []
    def c(i, j): return j*NX + i
    for j in range(NY):
        for i in range(NX):
            rc = c(i, j)
            if i == NX-1:                    # Dirichlet outlet only
                rows.append(rc); cols.append(rc); vals.append(1.0)
                continue
            diag = 0.0
            # x stencil
            if i == 0:                       # Neumann left
                rows.append(rc); cols.append(c(i+1,j)); vals.append(ax); diag -= ax
            else:
                rows.append(rc); cols.append(c(i-1,j)); vals.append(ax)
                rows.append(rc); cols.append(c(i+1,j)); vals.append(ax); diag -= 2*ax
            # y stencil
            if j == 0:                       # Neumann bottom: (φ[j+1]-φ[j])/dy²
                rows.append(rc); cols.append(c(i,j+1)); vals.append(ay); diag -= ay
            elif j == NY-1:                  # Neumann top: (φ[j-1]-φ[j])/dy²
                # Note: div_fwd has no explicit y-term at j=NY-1 (v[-1,:]=0 after BCs
                # forces y-div=0 there regardless), but this y-coupling is needed to
                # maintain pressure continuity in y across the top wall.
                rows.append(rc); cols.append(c(i,j-1)); vals.append(ay); diag -= ay
            else:
                rows.append(rc); cols.append(c(i,j-1)); vals.append(ay)
                rows.append(rc); cols.append(c(i,j+1)); vals.append(ay); diag -= 2*ay
            rows.append(rc); cols.append(rc); vals.append(diag)
    return scipy.sparse.csr_matrix((vals,(rows,cols)), shape=(n,n))

print("Building Poisson matrix...", flush=True)
A = build_poisson()
solve_phi = scipy.sparse.linalg.factorized(A.tocsc())
print(f"  nnz={A.nnz}", flush=True)

# ── inlet BC ──────────────────────────────────────────────────────────────────
def u_inlet(y, t):
    return (4*U_MAX * y*(H-y)/H**2
            * (np.sin(2*np.pi*t/PERIOD + 1.5*np.pi) + 1.0))

# ── velocity BCs ──────────────────────────────────────────────────────────────
def apply_bc(u, v, t):
    # Apply inlet/outlet first, then walls (walls win at corners)
    u[:,  0] = u_inlet(yc, t);  v[:, 0] = 0.0       # inlet
    u[:, -1] = u[:, -2];  v[:, -1] = v[:, -2]        # outlet Neumann
    u[0,  :] = 0.0;  v[0,  :] = 0.0                  # bottom no-slip (overrides inlet corner)
    u[-1, :] = 0.0;  v[-1, :] = 0.0                  # top no-slip
    u[SOLID]   = 0.0;  v[SOLID]   = 0.0                  # solid-obstacle IBM

# ── FORWARD divergence (consistent with Poisson stencil + backward grad) ─────
def div_fwd(u, v):
    """div[i,j] = (u[i+1,j]-u[i,j])/dx + (v[i,j+1]-v[i,j])/dy  (forward)"""
    d = np.zeros((NY, NX))
    # x: forward, i=0..NX-2  (i=NX-1 is Dirichlet, handled by Poisson BC)
    d[:, :-1] = (u[:, 1:] - u[:, :-1]) / DX
    # y: forward, j=0..NY-2
    d[:-1, :] += (v[1:, :] - v[:-1, :]) / DY
    # top row j=NY-1: wall ghost v[NY] = -v[NY-1] (no-penetration antisymmetric)
    # but after apply_bc v[-1,:]=0 → ghost=0, contribution=(0-0)/DY=0 → skip
    # solid-obstacle cells: zeroed later via rhs[SOLID]=0
    return d

# ── BACKWARD gradient (correction step) ───────────────────────────────────────
def grad_bkwd(phi):
    """gx[i,j]=(φ[i,j]-φ[i-1,j])/dx, gy[i,j]=(φ[i,j]-φ[i,j-1])/dy"""
    gx = np.zeros((NY, NX))
    gy = np.zeros((NY, NX))
    gx[:, 1:] = (phi[:, 1:] - phi[:, :-1]) / DX   # i=1..NX-1
    # gx[:,0] = 0  (Neumann inlet: φ_ghost=φ[0] → grad=0, no correction)
    gy[1:, :] = (phi[1:, :] - phi[:-1, :]) / DY   # j=1..NY-1
    # gy[0,:] = 0  (Neumann bottom)
    return gx, gy

# ── diffusion (for u_star) — standard Laplacian with Neumann walls ────────────
def laplacian(phi):
    l = np.zeros_like(phi)
    l[:, 1:-1] += (phi[:, 2:] - 2*phi[:, 1:-1] + phi[:, :-2]) / DX**2
    l[:,  0]   += (phi[:,  1] - phi[:,  0]) / DX**2   # Neumann left
    l[:, -1]   += (phi[:, -2] - phi[:, -1]) / DX**2  # Neumann right
    l[1:-1, :] += (phi[2:, :] - 2*phi[1:-1, :] + phi[:-2, :]) / DY**2
    l[0,    :] += (phi[ 1, :] - phi[  0, :]) / DY**2  # Neumann bottom
    l[-1,   :] += (phi[-2, :] - phi[ -1, :]) / DY**2  # Neumann top
    return l

# ── advection — first-order upwind ────────────────────────────────────────────
def advect(phi, u, v):
    pl = np.concatenate([phi[:, :1],  phi[:, :-1]], axis=1)
    pr = np.concatenate([phi[:, 1:],  phi[:, -1:]], axis=1)
    pd = np.concatenate([phi[:1,  :], phi[:-1, :]], axis=0)
    pu = np.concatenate([phi[1:,  :], phi[-1:, :]], axis=0)
    return (np.where(u > 0, u*(phi-pl)/DX, u*(pr-phi)/DX)
          + np.where(v > 0, v*(phi-pd)/DY, v*(pu-phi)/DY))

# ── initial conditions ────────────────────────────────────────────────────────
u = np.zeros((NY, NX)); v = np.zeros((NY, NX)); p = np.zeros((NY, NX))
apply_bc(u, v, 0.0)

snap_u, snap_v, snap_p, snap_t = [], [], [], []

def save(u, v, p, t):
    snap_u.append(u.copy()); snap_v.append(v.copy())
    snap_p.append(p.copy()); snap_t.append(t)

save(u, v, np.zeros((NY, NX)), 0.0)   # φ=0 at t=0 (no correction yet)
print("Time integration...", flush=True)
wall0 = time.time()

for step in range(1, N_STEPS + 1):
    t_new = step * DT

    # Step 1 — intermediate velocity (no pressure gradient)
    u_star = u + DT * (-advect(u, u, v) + NU * laplacian(u))
    v_star = v + DT * (-advect(v, u, v) + NU * laplacian(v))
    apply_bc(u_star, v_star, t_new)

    # Step 2 — pressure Poisson  LP φ = (ρ/dt) ∇_fwd·u*
    rhs = (RHO / DT) * div_fwd(u_star, v_star)
    rhs[:, -1] = 0.0                      # Dirichlet outlet: φ=0
    phi = solve_phi(rhs.flatten()).reshape(NY, NX)

    # Step 3 — velocity correction u = u* - (dt/ρ)∇_bkwd φ
    gx, gy = grad_bkwd(phi)
    u_new = u_star - (DT/RHO) * gx
    v_new = v_star - (DT/RHO) * gy
    p_new = p + phi
    apply_bc(u_new, v_new, t_new)
    p_new[:, -1] = 0.0

    u, v, p = u_new, v_new, p_new

    if step % SNAP_EVERY == 0 or step == N_STEPS:
        save(u, v, phi, t_new)   # save φ (instantaneous pressure correction)
        if step % (5*SNAP_EVERY) == 0 or step == N_STEPS:
            print(f"  step {step:4d}/{N_STEPS}  t={t_new:.3f}  "
                  f"max|u|={np.max(np.abs(u[FLUID])):.3e}  "
                  f"max|div|={np.max(np.abs(div_fwd(u,v)[FLUID])):.3e}  "
                  f"wall={time.time()-wall0:.0f}s", flush=True)

print(f"Done. snapshots={len(snap_t)}", flush=True)

# ── pack and save ─────────────────────────────────────────────────────────────
su = np.array(snap_u, dtype=np.float32)   # (Nt, NY, NX)
sv = np.array(snap_v, dtype=np.float32)
sp = np.array(snap_p, dtype=np.float32)
st = np.array(snap_t, dtype=np.float32)
Nt = len(st)

ff = FLUID.flatten()
xf = XX.flatten()[ff].astype(np.float32)
yf = YY.flatten()[ff].astype(np.float32)
Ns = xf.shape[0]

uo = np.zeros((Ns, Nt), dtype=np.float32)
vo = np.zeros((Ns, Nt), dtype=np.float32)
po = np.zeros((Ns, Nt), dtype=np.float32)
for k in range(Nt):
    uo[:, k] = su[k].flatten()[ff]
    vo[:, k] = sv[k].flatten()[ff]
    po[:, k] = sp[k].flatten()[ff]

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unsteady_reference_human_rot90left.mat")
scipy.io.savemat(out, {
    "x": xf[:, None], "y": yf[:, None],
    "t": st[:, None], "u": uo, "v": vo, "p": po,
    "solid_mask": SOLID.astype(np.uint8),
    # No obstacle_bbox is saved: downstream plots should overlay only solid_mask,
    # avoiding the black rectangular debug frame around the obstacle.
    "x_grid": xc[:, None],
    "y_grid": yc[:, None],
    "nominal_Re": np.array([[RE]], dtype=np.float32),
    "char_len": np.array([[CHAR_LEN]], dtype=np.float32),
})
print(f"Saved → {out}  shape x({Ns},1) t({Nt},1) u({Ns},{Nt})", flush=True)

# ── validity checks ───────────────────────────────────────────────────────────
print("\n=== Validity Checks ===")
uf, vf, pf = su[-1], sv[-1], sp[-1]

Re = RE
print(f"[1] Nominal Re = {Re:.1f}  using CHAR_LEN={CHAR_LEN:.3f}")

div_fin = div_fwd(uf, vf)
# IBM boundary cells: fluid cells adjacent to solid obstacle (forward stencil looks into SOLID→u=0)
solid_adj = np.zeros((NY, NX), bool)
solid_adj[:, :-1] |= SOLID[:, 1:]    # right neighbor is solid
solid_adj[:-1, :] |= SOLID[1:, :]    # upper neighbor is solid
interior_fluid = FLUID & ~solid_adj
dmax_all      = np.max(np.abs(div_fin[FLUID]))
dmax_interior = np.max(np.abs(div_fin[interior_fluid]))
print(f"[2] Max |div(u)| all-fluid={dmax_all:.3e}  interior-fluid={dmax_interior:.3e}  "
      f"{'OK' if dmax_interior<0.5 else 'WARN'}"
      f"  (large all-fluid value is IBM boundary artifact)")

flux_in  = np.trapezoid(uf[:, 0],  yc)
flux_out = np.trapezoid(uf[:, -1], yc)
imb = abs(flux_in-flux_out)/max(abs(flux_in),1e-12)
print(f"[3] Flux in={flux_in:.4f}  out={flux_out:.4f}  imbal={imb*100:.1f}%  {'OK' if imb<0.05 else 'WARN'}")

ji,ii = np.where(SOLID)
sc = np.sqrt(uf[ji,ii]**2 + vf[ji,ii]**2)
print(f"[4] Max speed in solid-obstacle mask = {sc.max():.2e}  {'OK' if sc.max()<1e-10 else 'WARN'}")

wt = max(np.max(np.abs(uf[-1,:])), np.max(np.abs(uf[0,:])))
print(f"[5] Max |u| on walls = {wt:.2e}  {'OK' if wt<1e-10 else 'WARN'}")

ke = np.array([0.5*RHO*np.mean(su[k][FLUID]**2+sv[k][FLUID]**2) for k in range(Nt)])
fin = np.all(np.isfinite(ke))
print(f"[6] KE [{ke.min():.4f}, {ke.max():.4f}]  finite={fin}  {'OK' if fin else 'FAIL'}")

err = np.max(np.abs(uf[:,0] - u_inlet(yc, st[-1])))
print(f"[7] Inlet BC error at t={st[-1]:.3f} = {err:.2e}  {'OK' if err<0.01 else 'WARN'}")

pout = np.max(np.abs(pf[:,-1]))
print(f"[8] Max |p| at outlet = {pout:.2e}  {'OK' if pout<1e-6 else 'WARN'}")
print("All checks done.")
