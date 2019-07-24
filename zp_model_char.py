
import torch
import torch.nn as nn
import os, sys, json, codecs

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from span_classifier import SpanClassifier


class BertZP(BertPreTrainedModel):
    def __init__(self, config, char2word, pro_num, max_relative_position):
        super(BertZP, self).__init__(config)
        print("zp_model_char.py: for model_type 'bert_char', 'char2word' not in use")
        assert type(pro_num) is int and pro_num > 1
        self.pro_num = pro_num
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.resolution_classifier = SpanClassifier(config.hidden_size, max_relative_position)
        self.detection_classifier = nn.Linear(config.hidden_size, 2)
        self.recovery_classifier = nn.Linear(config.hidden_size, pro_num)


    def forward(self, input_ids, mask, decision_mask,
            detection_refs, resolution_refs, recovery_refs, batch_type):
        char_repre, _ = self.bert(input_ids, None, mask, output_all_encoded_layers=False)
        char_repre = self.dropout(char_repre) # [batch, seq, dim]

        #detection
        detection_logits = self.detection_classifier(char_repre) # [batch, seq, 2]
        detection_outputs = detection_logits.argmax(dim=-1) # [batch, seq]
        detection_loss = torch.tensor(0.0)
        if detection_refs is not None:
            detection_loss = token_classification_loss(detection_logits, 2, detection_refs, decision_mask)

        #resolution
        if batch_type == 'resolution':
            resolution_start_logits, resolution_end_logits = self.resolution_classifier(char_repre, decision_mask)
            resolution_start_outputs = resolution_start_logits.argmax(dim=-1) # [batch, seq]
            resolution_end_outputs = resolution_end_logits.argmax(dim=-1) # [batch, seq]
            resolution_outputs = torch.stack([resolution_start_outputs, resolution_end_outputs], dim=-1) # [batch, seq, 2]
            resolution_loss = torch.tensor(0.0)
            if resolution_refs is not None: # [batch, seq, seq, 2]
                resolution_start_positions, resolution_end_positions = resolution_refs.split(1, dim=-1)
                resolution_start_positions = resolution_start_positions.squeeze(dim=-1) # [batch, seq, seq]
                resolution_end_positions = resolution_end_positions.squeeze(dim=-1) # [batch, seq, seq]
                resolution_loss = span_loss(resolution_start_logits, resolution_end_logits,
                        resolution_start_positions, resolution_end_positions, decision_mask)
                total_loss = detection_loss + resolution_loss
            return {'total_loss': total_loss, 'detection_loss': detection_loss, 'resolution_loss': resolution_loss}, \
                    detection_outputs, resolution_outputs

        #recovery
        if batch_type == 'recovery':
            recovery_logits = self.recovery_classifier(char_repre) # [batch, wordseq, pro_num]
            recovery_outputs = recovery_logits.argmax(dim=-1) # [batch, wordseq]
            recovery_loss = torch.tensor(0.0)
            if recovery_refs is not None:
                recovery_loss = token_classification_loss(recovery_logits, self.pro_num, recovery_refs, decision_mask)
                total_loss = detection_loss + recovery_loss
            return {'total_loss': total_loss, 'detection_loss': detection_loss, 'recovery_loss': recovery_loss}, \
                    detection_outputs, recovery_outputs

        assert False, "batch_type need to be either 'recovery' or 'resolution'"


def token_classification_loss(logits, num_labels, refs, masks): # [batch, seq, num_labels], scalar, [batch, 1]
    assert list(logits.size())[-1] == num_labels
    loss_fct = nn.CrossEntropyLoss()
    active_positions = masks.view(-1) == 1 # [batch*seq]
    active_logits = logits.view(-1,num_labels)[active_positions] # [batch*seq(sub), num_labels]
    active_refs = refs.view(-1)[active_positions] # [batch*seq(sub)]
    return loss_fct(active_logits, active_refs)


# logits: [batch, seq, seq]
# refs: [batch, seq, seq]
# masks: [batch, seq]
def token_classification_loss_v2(logits, refs, masks):
    num_tokens = torch.sum(masks).item()
    distrib = nn.Softmax(dim=-1)(logits)
    return -1.0 * torch.sum(distrib.log()*refs.float()) / num_tokens


# start_logits: [batch, seq, seq]
# end_logits: [batch, seq, seq]
# start_positions: [batch, seq, seq]
# end_positions: [batch, seq, seq]
# seq_masks: [batch, seq]
def span_loss(start_logits, end_logits, start_positions, end_positions, seq_masks):
    span_st_loss = token_classification_loss_v2(start_logits, start_positions, seq_masks)
    span_ed_loss = token_classification_loss_v2(end_logits, end_positions, seq_masks)
    return span_st_loss + span_ed_loss


