
import math
from typing import Callable
import torch
from torch.nn.functional import linear, normalize
from torch.nn import functional as F
from torch.nn import Parameter 
from torch import nn

class FullyConnected(nn.Module):
    def __init__(self, in_features, out_features, s=64.0, m=0.40, easy_margin=False):
        self.in_eatures = in_features
        self.out_features = out_features
        
        self.weight = Parameter(torch.FloatTensor(out_features, in_features))
        nn.init.xavier_uniform_(self.weight)


    def forward(self, input, label):
        # --------------------------- cos(theta) & phi(theta) ---------------------------
        
        return output