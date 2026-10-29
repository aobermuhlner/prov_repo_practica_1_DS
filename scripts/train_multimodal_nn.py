"""CPU/CUDA neural classifier; estimator classes remain importable for inference."""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.multimodal_nn import main

if __name__ == "__main__":
    main()
