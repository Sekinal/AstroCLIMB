"""Run an unchanged selected trainer/exporter with a portable model location."""
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from astroclimb.reproduction import main

if __name__ == "__main__":
    main()
