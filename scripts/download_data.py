"""Download the Fe-HCl-317 dataset from its original repository."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data import download_raw  # noqa: E402

if __name__ == "__main__":
    download_raw(force="--force" in sys.argv)
