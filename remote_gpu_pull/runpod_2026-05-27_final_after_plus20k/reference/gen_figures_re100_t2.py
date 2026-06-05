"""
Regenerate field-frame PNGs and animated GIF for the Re=100 T=2.0 s reference.
Matches the style of data/reference/figures/re100/ exactly:
  - rainbow colormap for all three fields
  - pcolormesh on the full NX×NY grid with cylinder cells masked white
  - Subplot titles: "u reference (Re=100)", "v reference (Re=100)",
                    "p (φ) reference (Re=100)"
  - Suptitle: "Re=100 reference  t = X.XXX s"
"""

import os
import numpy as np
import scipy.io
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from PIL import Image

# ── grid parameters (must match gen_unsteady_reference_re100.py) ──────────────
NX, NY = 220, 82
DX = 1.1 / NX
DY = 0.41 / NY
XC, YC, RC = 0.2, 0.2, 0.05

xc = (np.arange(NX) + 0.5) * DX   # cell centres x  (NX,)
yc = (np.arange(NY) + 0.5) * DY   # cell centres y  (NY,)
XX, YY = np.meshgrid(xc, yc)       # (NY, NX)
CYL = (XX - XC)**2 + (YY - YC)**2 <= RC**2

# pcolormesh needs node coordinates
xe = np.linspace(0.0, 1.1, NX + 1)
ye = np.linspace(0.0, 0.41, NY + 1)

# ── load mat file ─────────────────────────────────────────────────────────────
REPO = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
mat_path = os.path.join(REPO, "data", "reference", "unsteady_reference_re100_t2.mat")
out_dir  = os.path.join(REPO, "data", "reference", "figures", "re100_t2")
os.makedirs(out_dir, exist_ok=True)

print(f"Loading {mat_path} …")
data = scipy.io.loadmat(mat_path)
xf = data["x"].flatten()   # (Ns,)
yf = data["y"].flatten()
t_arr = data["t"].flatten()  # (Nt,)
U = data["u"]  # (Ns, Nt)
V = data["v"]
P = data["p"]
Ns, Nt = U.shape
print(f"  Ns={Ns}  Nt={Nt}  t=[{t_arr[0]:.3f}, {t_arr[-1]:.3f}]")

# Build index to map fluid points back to full NX×NY grid
idx_flat = np.round((xf / DX - 0.5)).astype(int) * 1 + np.round((yf / DY - 0.5)).astype(int) * NX
# Simpler: build via nearest-cell lookup
i_idx = np.clip(np.round(xf / DX - 0.5).astype(int), 0, NX - 1)
j_idx = np.clip(np.round(yf / DY - 0.5).astype(int), 0, NY - 1)

# ── colormap and limits (match old re100 style) ───────────────────────────────
CMAP = "rainbow"
VLIMS = {
    "u": (0.0,  1.0),
    "v": (-0.5, 0.5),
    "p": (-0.2, 3.0),
}

def make_frame(k):
    u_flat = U[:, k]
    v_flat = V[:, k]
    p_flat = P[:, k]

    # Fill full grid (NaN = cylinder)
    u_grid = np.full((NY, NX), np.nan)
    v_grid = np.full((NY, NX), np.nan)
    p_grid = np.full((NY, NX), np.nan)
    u_grid[j_idx, i_idx] = u_flat
    v_grid[j_idx, i_idx] = v_flat
    p_grid[j_idx, i_idx] = p_flat

    # Mask cylinder white
    u_masked = np.ma.masked_where(CYL, u_grid)
    v_masked = np.ma.masked_where(CYL, v_grid)
    p_masked = np.ma.masked_where(CYL, p_grid)

    fig, axes = plt.subplots(nrows=3, figsize=(6, 8))
    fig.suptitle(f"Re=100 reference  t = {t_arr[k]:.3f} s", fontsize=12)

    fields = [
        (u_masked, "u reference (Re=100)", VLIMS["u"]),
        (v_masked, "v reference (Re=100)", VLIMS["v"]),
        (p_masked, "p (φ) reference (Re=100)", VLIMS["p"]),
    ]
    for ax, (fld, title, (vmin, vmax)) in zip(axes, fields):
        cm = ax.pcolormesh(xe, ye, fld, cmap=CMAP, vmin=vmin, vmax=vmax,
                           shading="flat")
        cyl_patch = plt.Circle((XC, YC), RC, color="white", zorder=2)
        ax.add_patch(cyl_patch)
        ax.set_title(title, fontsize=9)
        ax.set_xlim(0.0, 1.1)
        ax.set_ylim(0.0, 0.41)
        ax.set_aspect("equal")
        fig.colorbar(cm, ax=ax, fraction=0.046, pad=0.04)

    plt.tight_layout()
    out_path = os.path.join(out_dir, f"field_frame_{k:03d}.png")
    plt.savefig(out_path, dpi=120)
    plt.close(fig)
    return out_path

# ── generate frames ───────────────────────────────────────────────────────────
paths = []
for k in range(Nt):
    p = make_frame(k)
    paths.append(p)
    if k % 10 == 0:
        print(f"  frame {k}/{Nt-1}  t={t_arr[k]:.3f}", flush=True)
print(f"Wrote {len(paths)} frames to {out_dir}")

# ── assemble GIF ──────────────────────────────────────────────────────────────
gif_path = os.path.join(REPO, "data", "reference", "figures", "re100_t2_animation.gif")
frames = [Image.open(p) for p in paths]
frames[0].save(
    gif_path,
    save_all=True,
    append_images=frames[1:],
    duration=120,
    loop=0,
)
print(f"GIF → {gif_path}")
