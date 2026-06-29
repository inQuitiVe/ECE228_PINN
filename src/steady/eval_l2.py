"""Load checkpointed weights and compute relative L2 error vs Fluent.
Usage: python3 eval_l2.py <module> <weights.pickle>
  <module> = Steady_ref  or  Steady_adpative
"""
import sys, numpy as np
mod_name, wpath = sys.argv[1], sys.argv[2]
mod = __import__(mod_name)
PINN = mod.PINN_laminar_flow
relative_l2_error = mod.relative_l2_error

lb = np.array([0.0, 0.0]); ub = np.array([1.1, 0.41])
uv_layers = [2] + 8 * [40] + [5]

# Minimal dummy point sets — only the loaded network weights matter for predict.
Collo  = np.array([[0.5, 0.2]])
INLET  = np.array([[0.0, 0.2, 1.0, 0.0]])
OUTLET = np.array([[1.1, 0.2]])
WALL   = np.array([[0.2, 0.25]])

model = PINN(Collo, INLET, OUTLET, WALL, uv_layers, lb, ub,
             ExistModel=1, uvDir=wpath)
errs = relative_l2_error(model, ref_mat_path='FluentReferenceMu002/FluentSol.mat',
                         r_cyl=0.05, xc=0.2, yc=0.2)
print("EVAL_L2_RESULT", errs['u'], errs['v'], errs['p'])
