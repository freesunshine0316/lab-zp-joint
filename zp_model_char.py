
import torch
import torch.nn as nn
import os, sys, json, codecs

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from span_classifier import SSAClassifier


class ZPBertChar(BertPreTrainedModel):
    def __init__(self, config, pro_num=-1):
        super(BertZeroProMTLChar, self).__init__(config)
        assert pro_num > 0
        self.pro_num = pro_num
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.resolution_classifier = SSAClassifier()
        self.detection_classifier = nn.Linear(config.hidden_size, 2)
        self.recovery_classifier = nn.Linear(config.hidden_size, pro_num)


    def forward(self, input_ids, mask, decision_mask,
            detection_refs, resolution_refs, recovery_refs, batch_type):
        char_repre, _ = self.bert(input_ids, None, mask, output_all_encoded_layers=False)
        char_repre = self.dropout(char_repre) # [batch, seq, dim]

        batch_size, seq_num, hidden_dim = list(char_repre.size())
        #detection
        detection_logits = self.detection_classifier(char_repre) # [batch, seq, 2]
        detection_outputs = detection_logits.argmax(dim=-1) # [batch, seq]
        if detection_refs is not None:
            detection_loss = token_classification_loss(detection_logits, 2, detection_refs, word_mask)
            if

        #resolution
        if batch_type == 'resolution':
            assert False, 'under construction'
            resolution_logits_st, resolution_logits_ed = self.resolution_classifier(word_repre)
            tmp1 = word_mask.unsqueeze(1) # [batch, 1, wordseq]
            tmp2 = tmp1.transpose(1, 2) # [batch, wordseq, 1]
            square_mask = tmp2.matmul(tmp1) # [batch, wordseq, wordseq]
            return

        #recovery
        if batch_type == 'recovery':
            recovery_logits = self.recovery_classifier(word_repre) # [batch, wordseq, pro_num]
            recovery_outputs = recovery_logits.argmax(dim=-1) # [batch, wordseq]
            if recovery_refs is not None:
                recovery_loss = token_classification_loss(recovery_logits, self.pro_num, recovery_refs, word_mask)
                assert detection_refs is not None
                return detection_loss + recovery_loss, detection_outputs, recovery_outputs
            else:
                return None, detection_outputs, recovery_outputs

        assert False, "batch_type need to be either 'recovery' or 'resolution'"


def token_classification_loss(logits, num_labels, refs, masks): # [batch, seq, num_labels], scalar, [batch, 1]
    loss_fct = nn.CrossEntropyLoss()
    active_positions = masks.view(-1) == 1 # [batch*seq]
    active_logits = logits.view(-1,num_labels)[active_positions] # [batch*seq(sub), num_labels]
    active_refs = refs.view(-1)[active_positions] # [batch*seq(sub)]
    return loss_fct(active_logits, active_refs)


# start_logits: [batch, seq, seq, dim]
# end_logits: [batch, seq, seq, dim]
# start_positions: [batch, seq]
# end_positions: [batch, seq]
# self_mask: [batch, seq, seq]
def span_loss(start_logits, end_logits, start_positions, end_positions, self_mask):
    pass
