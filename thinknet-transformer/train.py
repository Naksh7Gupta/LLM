import sys
import os

# Add the directory containing configs/model/training/etc to sys.path
project_root = os.path.dirname(os.path.abspath(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

from training.trainer import train

if __name__ == "__main__":
    train()
