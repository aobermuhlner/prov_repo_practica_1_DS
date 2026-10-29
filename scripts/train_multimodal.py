"""CLI entrypoint; estimator classes stay importable when loading joblib files."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.multimodal_training import main

if __name__ == "__main__":
    main()
