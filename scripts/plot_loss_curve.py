import os
import pickle
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Resolve data/output paths relative to this file so the script runs from anywhere.
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
EXP = os.path.join(ROOT, 'results', 'experiments', 'steady_adaptive_sweep')
# Heavy loss pickles are not tracked; place the run outputs here to regenerate the figure.
A = pickle.load(open(os.path.join(EXP, 'logs', 'adaptive_loss.pickle'), 'rb'))
F = pickle.load(open(os.path.join(EXP, 'logs', 'fixed_loss.pickle'), 'rb'))
la = [float(x) for x in A['loss']]; na = A.get('n_adam')
lf = [float(x) for x in F['loss']]; nf = F.get('n_adam')

plt.figure(figsize=(9, 5))
plt.semilogy(range(len(la)), la, color='tab:blue',  lw=1.0,
             label=f'adaptive  (final {la[-1]:.2e})')
plt.semilogy(range(len(lf)), lf, color='tab:orange', lw=1.0,
             label=f'fixed  (final {lf[-1]:.2e})')
if na:
    plt.axvline(na, color='tab:blue',  ls='--', alpha=0.5)
if nf:
    plt.axvline(nf, color='tab:orange', ls='--', alpha=0.5,
                label='Adam → L-BFGS')
plt.ylim(1e-4, 5.0)   # clip the one-point L-BFGS start transient (~1e5)
plt.xlabel('training step  (Adam + L-BFGS)')
plt.ylabel('loss')
plt.title('Loss history: adaptive vs fixed collocation (20k Adam + 50k L-BFGS)')
plt.legend(); plt.grid(True, alpha=0.3, which='both')
plt.tight_layout()
os.makedirs(os.path.join(EXP, 'figures'), exist_ok=True)
plt.savefig(os.path.join(EXP, 'figures', 'full_loss_curve.png'), dpi=150)
print('saved full_loss_curve.png  adaptive_len=%d fixed_len=%d' % (len(la), len(lf)))
