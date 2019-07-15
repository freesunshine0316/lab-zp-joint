
import os, sys, json, codecs
import torch.nn as nn
from multi_headed_attn import MultiHeadedAttention


# span classifier based on self-attention
class SpanClassifier(nn.Module):
    def __init__(self, hidden_dim, max_relative_position):
        super(SpanClassifier, self).__init__()
        self.layer_norm = nn.LayerNorm(hidden_dim, eps=1e-6)
        self.span_st_attn = MultiHeadedAttention(1, hidden_dim, max_relative_positions=max_relative_position)
        self.span_ed_attn = MultiHeadedAttention(1, hidden_dim, max_relative_positions=max_relative_position)
        if max_relative_position > 0.0:
            print("Setting max_relative_position to {}".format(max_relative_position))

    def forward(self, repre, mask):
        #repre = self.layer_norm(repre)

        tmp1 = mask.unsqueeze(1) # [batch, 1, seq]
        tmp2 = tmp1.transpose(1, 2) # [batch, seq, 1]
        square_mask = tmp2.matmul(tmp1).byte() # [batch, seq, seq]
        square_mask = ~square_mask

        span_st_logits = self.span_st_attn(repre, repre, repre,
                mask=square_mask, type="self") # [batch, seq, seq]
        span_ed_logits = self.span_ed_attn(repre, repre, repre,
                mask=square_mask, type="self") # [batch, seq, seq]
        return span_st_logits, span_ed_logits


