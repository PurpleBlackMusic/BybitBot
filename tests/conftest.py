import os
import sys
from pathlib import Path

os.environ.setdefault("BYBITBOT_ENV", "test")
os.environ.setdefault("BYBITBOT_SKIP_INTEGRITY", "1")
os.environ.setdefault("BYBITBOT_DISABLE_ENV_FILE", "1")

# Ensure the project root is on sys.path for module imports
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))
