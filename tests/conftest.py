import os
import sys
from pathlib import Path

os.environ.setdefault("BYBITBOT_ENV", "test")

# Ensure the project root is on sys.path for module imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
