__version__ = "0.5.5"

import sys
from pathlib import Path
# Add benchmarks/ directory to sys.path, so we can import guacamol from the root directory
sys.path.append(str(Path(__file__).resolve().parent.parent))

from guacamol.assess_goal_directed_generation import (
    assess_goal_directed_generation, 
    assess_goal_directed_from_smiles
)