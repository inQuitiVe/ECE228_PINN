import os
import sys


ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STEADY_DIR = os.path.join(ROOT, "PINN_steady")
if STEADY_DIR not in sys.path:
    sys.path.insert(0, STEADY_DIR)

from steady_flow_cylinder_pytorch import main


if __name__ == "__main__":
    main()
