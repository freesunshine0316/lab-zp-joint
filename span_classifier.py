
import os, sys, json, codecs
import torch.nn as nn
from multi_headed_attn import MultiHeadedAttention


# span classifier based on multi-head self-attention
class SpanClassifier(nn.Module):
    def __init__(self, hidden_dim):
        super(SpanClassifier, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.span_st_pred = MultiHeadedAttention(1, hidden_dim)
        self.span_ed_pred = MultiHeadedAttention(1, hidden_dim)

    def forward(self, repre, mask):
        repre_norm = self.layer_norm(repre)
        span_st_dist = self.span_st_pred(repre_norm, repre_norm, repre_norm,
                mask=mask, attn_type="self") # [batch, seq, seq]
        span_ed_dist = self.span_ed_pred(repre_norm, repre_norm, repre_norm,
                mask=mask, attn_type="self") # [batch, seq, seq]
        return span_st_dist, span_ed_dist
