import os
import csv
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Resolve data/output paths relative to this file so the script runs from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, 'results', 'experiments', 'steady_adaptive_sweep')

rows = []
with open(os.path.join(EXP, 'benchmarks', 'summary.csv')) as f:
    for r in csv.DictReader(f):
        rows.append(r)

def get(name):
    for r in rows:
        if r['variant'] == name:
            return r
    return None

def vals(names, key):
    return [float(get(n)[key]) for n in names]

BASE = 'baseline'
# axis groupings (variant names in ascending x order), and their x values
groups = {
    'n_seed_neighbors': (['nseed_1','nseed_2','baseline','nseed_10'],
                         [1,2,5,10], 5),
    'accept_quantile':  (['acceptq_090','baseline','acceptq_097','acceptq_099'],
                         [0.90,0.95,0.97,0.99], 0.95),
    'n_exploration_pts':(['nexplore_500','baseline','nexplore_2000','nexplore_5000','nexplore_10000'],
                         [500,1000,2000,5000,10000], 1000),
}

# ---------- Figure 1: final loss vs each parameter ----------
xlabels = {
    'n_seed_neighbors':  '# of neighbors explored',
    'accept_quantile':   'Accepted quantile',
    'n_exploration_pts': '# of global exploration points',
}
# Loss figure drops the last exploration point (nexplore_10000) per request.
loss_groups = dict(groups)
_n, _x, _b = groups['n_exploration_pts']
loss_groups['n_exploration_pts'] = (_n[:-1], _x[:-1], _b)

fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
for ax, (pname, (names, xs, base_x)) in zip(axes, loss_groups.items()):
    ys = vals(names, 'loss_after_lbfgs')
    ax.plot(xs, ys, 'o-', color='steelblue', lw=2, ms=7)
    ax.set_xlabel(xlabels[pname]); ax.set_ylabel('Final loss')
    ax.grid(True, alpha=0.3)
    if pname == 'n_exploration_pts':
        ax.set_xscale('log')
plt.tight_layout()
os.makedirs(os.path.join(EXP, 'figures'), exist_ok=True)
plt.savefig(os.path.join(EXP, 'figures', 'sweep_loss.png'), dpi=150)
plt.close()

# ---------- Figure 2: relative L2 errors vs each parameter ----------
fig, axes = plt.subplots(1, 3, figsize=(15, 4.2))
colors = {'err_u':'tab:blue','err_v':'tab:green','err_p':'tab:red'}
labels = {'err_u':'u','err_v':'v','err_p':'p'}
for ax, (pname, (names, xs, base_x)) in zip(axes, groups.items()):
    for key in ('err_u','err_v','err_p'):
        ys = vals(names, key)
        ax.plot(xs, ys, 'o-', color=colors[key], lw=2, ms=6, label=labels[key])
    ax.axvline(base_x, color='crimson', ls='--', alpha=0.5, label='baseline x')
    ax.set_xlabel(pname); ax.set_ylabel('relative L2 error')
    ax.set_title(f'L2 error vs {pname}')
    ax.grid(True, alpha=0.3); ax.legend(fontsize=8)
    if pname == 'n_exploration_pts':
        ax.set_xscale('log')
plt.tight_layout()
plt.savefig(os.path.join(EXP, 'figures', 'sweep_l2error.png'), dpi=150)
plt.close()
print('wrote sweep_loss.png and sweep_l2error.png')
