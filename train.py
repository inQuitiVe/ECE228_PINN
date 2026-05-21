import os
import sys


ROOT = os.path.dirname(os.path.abspath(__file__))
SRC = os.path.join(ROOT, "src")
if SRC not in sys.path:
    sys.path.insert(0, SRC)

from pinn_laminar_flow import steady, unsteady


def main():
    if len(sys.argv) < 2 or sys.argv[1] not in {"steady", "unsteady"}:
        print("Usage: python3 train.py {steady|unsteady} [training options]", file=sys.stderr)
        print("Example: python3 train.py steady --device mps", file=sys.stderr)
        raise SystemExit(2)

    mode = sys.argv.pop(1)
    if mode == "steady":
        steady.main()
    else:
        unsteady.main()


if __name__ == "__main__":
    main()
