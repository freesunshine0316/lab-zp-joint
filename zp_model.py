
import torch
import torch.nn as nn
import os, sys, json, codecs

from pytorch_pretrained_bert.modeling import BertPreTrainedModel, BertModel
from span_classifier import SpanClassifier


class BertZP(BertPreTrainedModel):
    def __init__(self, config, char2word, pro_num):
        super(BertZP, self).__init__(config)
        assert pro_num > 0
        self.pro_num = pro_num
        self.char2word = char2word
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        self.resolution_classifier = SpanClassifier(config.hidden_size)
        self.detection_classifier = nn.Linear(config.hidden_size, 2)
        self.recovery_classifier = nn.Linear(config.hidden_size, pro_num)


    def forward(self, input_ids, mask, word_mask, input_char2word, input_char2word_mask,
            detection_refs, resolution_refs, recovery_refs, batch_type):
        char_repre, _ = self.bert(input_ids, None, mask, output_all_encoded_layers=False)
        char_repre = self.dropout(char_repre) # [batch, seq, dim]

        # cast from char-level to word-level
        batch_size, seq_num, hidden_dim = list(char_repre.size())
        _, wordseq_num, word_len = list(input_char2word.size())
        if self.char2word in ('first', 'last', ):
            assert word_len == 1
        offset = torch.arange(batch_size).view(batch_size, 1, 1).expand(-1, wordseq_num, word_len) * seq_num
        if torch.cuda.is_available():
            offset = offset.cuda()
        positions = (input_char2word + offset).view(batch_size * wordseq_num * word_len)
        word_repre = torch.index_select(char_repre.contiguous().view(batch_size * seq_num, hidden_dim), 0, positions)
        word_repre = word_repre.view(batch_size, wordseq_num, word_len, hidden_dim)
        word_repre = word_repre * input_char2word_mask.unsqueeze(-1)
        # word_repre: [batch, wordseq, dim]
        word_repre = word_repre.mean(dim=2) if self.char2word == 'mean' else word_repre.sum(dim=2)

        #detection
        detection_logits = self.detection_classifier(word_repre) # [batch, wordseq, 2]
        detection_outputs = detection_logits.argmax(dim=-1) # [batch, wordseq]
        if detection_refs is not None:
            detection_loss = classification_loss(detection_logits, detection_refs, word_mask, 2)

        #resolution
        if batch_type == 'resolution':
            resolution_start_logits, resolution_end_logits = self.resolution_classifier(word_repre, word_mask)
            resolution_start_outputs = resolution_start_logits.argmax(dim=-1)
            resolution_end_outputs = resolution_end_logits.argmax(dim=-1)
            resolution_outputs = torch.stack([resolution_start_outputs, resolution_end_outputs], dim=-1) # [batch, wordseq, 2]
            if resolution_refs is not None:
                #TODO: (1) char model part (2) data stream part
                resolution_start_positions, resolution_end_positions = resolution_refs.split(1, dim=2)
                resolution_start_positions = resolution_start_positions.squeeze(dim=2)
                resolution_start_positions = resolution_start_positions.squeeze(dim=2)
                resolution_loss = span_loss(resolution_start_logits, resolution_end_logits,
                        resolution_start_positions, resolution_end_positions, word_mask)
                assert detection_refs is not None
                return detection_loss + resolution_loss, detection_outputs, resolution_outputs
            else:
                return None, detection_outputs, resolution_outputs

        #recovery
        if batch_type == 'recovery':
            recovery_logits = self.recovery_classifier(word_repre) # [batch, wordseq, pro_num]
            recovery_outputs = recovery_logits.argmax(dim=-1) # [batch, wordseq]
            if recovery_refs is not None:
                recovery_loss = classification_loss(recovery_logits, recovery_refs, word_mask, self.pro_num)
                assert detection_refs is not None
                return detection_loss + recovery_loss, detection_outputs, recovery_outputs
            else:
                return None, detection_outputs, recovery_outputs

        assert False, "batch_type need to be either 'recovery' or 'resolution'"


# logits: [batch, seq, num_labels]
# refs: [batch, seq]
# seq_masks: [batch, seq]
def classification_loss(logits, refs, seq_masks, num_labels):
    loss_fct = nn.CrossEntropyLoss()
    active_positions = seq_masks.view(-1) == 1 # [batch*seq]
    active_logits = logits.view(-1,num_labels)[active_positions] # [batch*seq(sub), num_labels]
    active_refs = refs.view(-1)[active_positions] # [batch*seq(sub)]
    return loss_fct(active_logits, active_refs)


# start_logits: [batch, seq, seq]
# end_logits: [batch, seq, seq]
# start_positions: [batch, seq]
# end_positions: [batch, seq]
# seq_masks: [batch, seq]
def span_loss(start_logits, end_logits, start_positions, end_positions, seq_masks):
    num_labels = list(seq_masks.size())[1]
    span_st_loss = classification_loss(start_logits, start_positions, seq_masks, num_labels)
    span_ed_loss = classification_loss(end_logits, end_positions, seq_masks, num_labels)
    return span_st_loss + span_ed_loss


