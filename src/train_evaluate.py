"""
Delegates training and evaluation pipeline from src package.
"""
import sys
from pathlib import Path

# Add project root to path
root_dir = Path(__file__).resolve().parent.parent
if str(root_dir) not in sys.path:
    sys.path.insert(0, str(root_dir))

from train_evaluate import run_training_pipeline

if __name__ == "__main__":
    run_training_pipeline()
