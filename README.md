# PINN-laminar-flow
Physics-informed neural network (PINN) for solving fluid dynamics problems

# Reference paper
This repo include the implementation of mixed-form physics-informed neural networks in paper: 

[Chengping Rao, Hao Sun and Yang Liu. Physics-informed deep learning for incompressible laminar flows.](https://arxiv.org/abs/2002.10558)

- This paper has been published by TAML, those who has access to Elsevier database can refer to https://www.sciencedirect.com/science/article/pii/S2095034920300350 for camera-ready version. 

# Description for each folder
- **FluentReferenceMu002**: Reference solution from Ansys Fluent for steady flow;
<!--- - **FluentReferenceUnsteady**: Reference solution from Ansys Fluent for unsteady flow; --->
- **PINN_steady**: Implementation for steady flow with PINN;
- **PINN_unsteady**: Implementation for unsteady flow with PINN;

# Results overview

![](https://github.com/Raocp/PINN-laminar-flow/blob/master/PINN_steady/uvp.png)

> Steady flow past a cylinder (left: physics-informed neural network; right: Ansys Fluent.)


![](https://github.com/Raocp/PINN-laminar-flow/blob/master/PINN_unsteady/uvp_animation.gif)

> Transient flow past a cylinder (physics-informed neural network result)

# Note
- These implementations were developed and tested on the GPU version of TensorFlow 1.10.0. 
- The original scripts in `PINN_steady/SteadyFlowCylinder_mixed.py` and `PINN_unsteady/TransientFlowCylinder.py` are still TensorFlow 1.x based.

# PyTorch migration
- A first PyTorch port of the steady cylinder case is available at `PINN_steady/steady_flow_cylinder_pytorch.py`.
- Install the PyTorch-side dependencies with `pip install -r requirements-pytorch.txt`.
- Run the steady PyTorch solver with `python3 PINN_steady/steady_flow_cylinder_pytorch.py`.
- The new script writes a PyTorch checkpoint (`uvNN_torch.pt` by default), loss history (`loss_history_torch.pickle`), and a comparison figure (`uvp_torch.png`).
- The transient solver has not been ported yet and still uses the TensorFlow implementation.
