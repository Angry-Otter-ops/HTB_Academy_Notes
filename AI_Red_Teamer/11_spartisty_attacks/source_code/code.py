import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# Import utilities from HTB Evasion Library
from htb_ai_library.utils import (
    set_reproducibility,
    save_model,
    load_model,
    HTB_GREEN,
    NODE_BLACK,
    HACKER_GREY,
    WHITE,
    AZURE,
    NUGGET_YELLOW,
    MALWARE_RED,
    VIVID_PURPLE,
    AQUAMARINE,
)
from htb_ai_library.data import get_mnist_loaders
from htb_ai_library.models import MNISTClassifierWithDropout
from htb_ai_library.training import train_model, evaluate_accuracy
from htb_ai_library.visualization import use_htb_style

# Apply HTB theme globally to all plots
use_htb_style()

# Set reproducibility
set_reproducibility(1337)

# Configure device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")
if device.type == "cuda":
    print(f"GPU: {torch.cuda.get_device_name(0)}")
