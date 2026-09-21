"""Enter the package without changing cwd or exporting PYTHONPATH to user tools."""

import runpy
import sys


if __name__ == "__main__":
    module = "bridge"
    if sys.argv[1:2] == ["--desktop-launch"]:
        del sys.argv[1]
        module = "bridge.launch"
    runpy.run_module(module, run_name="__main__")
