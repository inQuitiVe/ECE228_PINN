import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UNSTEADY_DIR = os.path.join(ROOT, "PINN_unsteady")
if UNSTEADY_DIR not in sys.path:
    sys.path.insert(0, UNSTEADY_DIR)

from transient_flow_cylinder_pytorch import main


if __name__ == "__main__":
    main()
