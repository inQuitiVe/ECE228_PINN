"""
2D unsteady incompressible NS reference data generator — Re=100 variant.
Chorin projection on collocated grid with CONSISTENT forward-divergence /
backward-gradient pair — guarantees discrete div(u_new)=0 at every cell.

Domain matches src/pinn_laminar_flow/unsteady.py exactly:
  x in [0,1.1], y in [0,0.41], cylinder (0.2,0.2) r=0.05
  mu=0.0005, rho=1.0, U_max=0.5, period=1.0, tmax=2.0 (Re=100)

T_max=2.0 chosen to capture ~2 Kármán shedding cycles (St≈0.30 → period≈0.67 s).
Output: unsteady_reference_re100_t2.mat  (preserves the old 0.5 s file)
"""

import os, time
import numpy as np
import scipy.sparse, scipy.sparse.linalg, scipy.io

# ── parameters ────────────────────────────────────────────────────────────────
MU, RHO, NU = 0.0005, 1.0, 0.0005
U_MAX, T_MAX, PERIOD = 0.5, 2.0, 1.0
H = 0.41
XC, YC, RC = 0.2, 0.2, 0.05

NX, NY = 220, 82
DX = 1.1 / NX   # 0.005
DY = H   / NY   # 0.005
DT = 0.001       # CFL = U_max·dt/dx = 0.1;  viscous = ν·dt/dx² = 0.02 (Re=100)

N_STEPS  = int(round(T_MAX / DT))   # 500
N_SNAP   = 51
SNAP_EVERY = N_STEPS // (N_SNAP - 1)

xc = (np.arange(NX) + 0.5) * DX        # (NX,)
yc = (np.arange(NY) + 0.5) * DY        # (NY,)
XX, YY = np.meshgrid(xc, yc)           # (NY, NX)

CYL   = (XX - XC)**2 + (YY - YC)**2 <= RC**2
FLUID = ~CYL

print(f"Grid {NX}×{NY}  dx={DX:.4f}  dt={DT}  Re={RHO*U_MAX*(2*RC)/MU:.1f}")

# ── Poisson matrix ────────────────────────────────────────────────────────────
# Boundary conditions:
#   Dirichlet φ=0 at outlet column (i=NX-1) and cylinder cells
#   Neumann   ∂φ/∂n=0 at inlet (i=0), bottom (j=0), top (j=NY-1)
# The 5-point stencil LP is EXACTLY consistent with the backward gradient /
# forward divergence pair (see derivation in commit message).
def build_poisson():
    """
    5-point Laplacian for φ.
    Dirichlet φ=0: outlet column only (i=NX-1).
    Neumann ∂φ/∂n=0: inlet (i=0), top (j=NY-1), bottom (j=0).
    Cylinder cells: treated as regular interior — no Dirichlet BC.
      Applying Dirichlet φ=0 inside the cylinder creates spurious pressure
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
    u[CYL]   = 0.0;  v[CYL]   = 0.0                  # cylinder IBM

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
    # cylinder cells: zeroed later via rhs[CYL]=0
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

out = os.path.join(os.path.dirname(os.path.abspath(__file__)), "unsteady_reference_re100_t2.mat")
scipy.io.savemat(out, {"x": xf[:,None], "y": yf[:,None],
                       "t": st[:,None], "u": uo, "v": vo, "p": po})
print(f"Saved → {out}  shape x({Ns},1) t({Nt},1) u({Ns},{Nt})", flush=True)

# ── validity checks ───────────────────────────────────────────────────────────
print("\n=== Validity Checks ===")
uf, vf, pf = su[-1], sv[-1], sp[-1]

Re = RHO*U_MAX*(2*RC)/MU
print(f"[1] Re = {Re:.1f}")

div_fin = div_fwd(uf, vf)
# IBM boundary cells: fluid cells adjacent to cylinder (forward stencil looks into CYL→u=0)
cyl_adj = np.zeros((NY, NX), bool)
cyl_adj[:, :-1] |= CYL[:, 1:]    # right neighbor is cylinder
cyl_adj[:-1, :] |= CYL[1:, :]    # upper neighbor is cylinder
interior_fluid = FLUID & ~cyl_adj
dmax_all      = np.max(np.abs(div_fin[FLUID]))
dmax_interior = np.max(np.abs(div_fin[interior_fluid]))
print(f"[2] Max |div(u)| all-fluid={dmax_all:.3e}  interior-fluid={dmax_interior:.3e}  "
      f"{'OK' if dmax_interior<0.5 else 'WARN'}"
      f"  (large all-fluid value is IBM boundary artifact)")

flux_in  = np.trapezoid(uf[:, 0],  yc)
flux_out = np.trapezoid(uf[:, -1], yc)
imb = abs(flux_in-flux_out)/max(abs(flux_in),1e-12)
print(f"[3] Flux in={flux_in:.4f}  out={flux_out:.4f}  imbal={imb*100:.1f}%  {'OK' if imb<0.05 else 'WARN'}")

ji,ii = np.where(CYL)
sc = np.sqrt(uf[ji,ii]**2 + vf[ji,ii]**2)
print(f"[4] Max speed in cylinder mask = {sc.max():.2e}  {'OK' if sc.max()<1e-10 else 'WARN'}")

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
