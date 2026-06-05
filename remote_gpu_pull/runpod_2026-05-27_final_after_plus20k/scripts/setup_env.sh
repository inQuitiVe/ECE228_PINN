#!/usr/bin/env bash
# Setup Python environment for ECE228-PINN on RunPod (CUDA) or Mac (MPS/CPU).
# Usage: bash scripts/setup_env.sh

set -e

echo "=== Installing Python dependencies ==="
pip install -q -r requirements.txt

# pyDOE2 is installed but the code imports 'pyDOE' — create a shim.
PYDOE_SHIM=$(python3 -c "
import sys, os
site = [p for p in sys.path if 'dist-packages' in p or 'site-packages' in p]
print(os.path.join(site[0], 'pyDOE.py'))
")

if [ ! -f "$PYDOE_SHIM" ]; then
    echo "from pyDOE2 import *" > "$PYDOE_SHIM"
    echo "=== Created pyDOE shim at $PYDOE_SHIM ==="
else
    echo "=== pyDOE shim already exists at $PYDOE_SHIM ==="
fi

echo "=== Setup complete. Test with: python3 -c 'import torch, pyDOE, scipy, matplotlib; print(torch.__version__)' ==="
