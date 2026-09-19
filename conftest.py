"""Ensures the project root is importable as the ``src`` package during tests.

Its presence marks the pytest rootdir; the explicit insert keeps imports working
regardless of the working directory pytest is launched from.
"""

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
