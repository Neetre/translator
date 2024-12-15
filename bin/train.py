from dataclasses import dataclass
import torch
import torch.nn as nn
import torch.nn.functional as F

from data_loader import MTDataset, MTConfig

@dataclass
class Hyper:
    pass