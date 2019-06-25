
import os, sys, json, codecs
import torch.nn as nn
from multi_headed_attn import MultiHeadedAttention


# self-span attention
class SSAClassifier(nn.Module):
    def __init__(self):
        super(SSAClassifier, self).__init__()

    def forward(self):
        pass
