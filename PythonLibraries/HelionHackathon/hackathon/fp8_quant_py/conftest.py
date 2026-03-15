"""conftest.py — pytest path setup for fp8_quant_py tests.

Adds the parent helion problems directory to sys.path so that
`from utils import ...` in reference.py resolves correctly when
running pytest from within fp8_quant_py/.
"""
import sys
from pathlib import Path

# .../problems/helion/fp8_quant_py  →  .../problems/helion
_HELION_DIR = Path(__file__).parent.parent
if str(_HELION_DIR) not in sys.path:
    sys.path.insert(0, str(_HELION_DIR))
