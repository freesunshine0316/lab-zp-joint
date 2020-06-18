
import os, sys, json
import argparse
import numpy as np
import time
import random

import torch
import torch.nn as nn

import config_utils

from pytorch_pretrained_bert.tokenization import BertTokenizer

from flask import Flask, request
from flask_restful import Api, Resource, reqparse


app = Flask(__name__)
api = Api(app)


class ZPRecover(Resource):
    def put(self):
        # load data and make_batches
        data_type = 'recovery_inference'
        data = make_data_recovery(request.form['conversation'], tokenizer)
        features = zp_datastream.extract_features(data, tokenizer,
                char2word=FLAGS.char2word, data_type=data_type, is_only_azp=False)
        batches = zp_datastream.make_batch(data_type, features, FLAGS.batch_size,
                is_sort=False, is_shuffle=False)
        decisions = inference(model, FLAGS.model_type, batches, pro_mapping)
        # TODO: post processing and return
        return


def make_data_recovery(conversation, tokenizer):
    data = {'sentences': [], # [batch, wordseq]
            'sentences_bert_idxs': [], # [batch, wordseq, wordlen]
            'sentences_bert_toks': [], # [batch, seq]
            'zp_info': []} # [a sequence of ...]

    sent = ['[CLS]',] # [wordseq]
    sent_bert_idxs = [[0],] # [wordseq, wordlen]
    sent_bert_toks = ['[CLS]',] # [seq]
    j = 1
    j_char = 1
    sent_nps = []
    for i, sentence in enumerate(conversation):
        old_j = j
        sentence = sentence.strip().split()
        for word in sentence:
            sent.append(word)
            j += 1
            sent_bert_idxs.append([])
            for char in tokenizer.tokenize(word):
                sent_bert_idxs[-1].append(len(sent_bert_toks))
                sent_bert_toks.append(char)
                j_char += 1
            if len(sent_bert_idxs[-1]) == 0:
                sent.pop()
                j -= 1
                sent_bert_idxs.pop()

    data['sentences'].append(sent)
    data['sentences_bert_idxs'].append(sent_bert_idxs)
    data['sentences_bert_toks'].append(sent_bert_toks)

    return data


def inference(model, model_type, batches, pro_mapping):
    model.eval()
    N = 0
    recovery_decisions = []
    for step, ori_batch in enumerate(batches):
        # execution
        batch = {k: v.to(device) if type(v) == torch.Tensor else v for k, v in ori_batch.items()}
        step_loss, step_out = forward_step(model, model_type, batch)
        recovery_out = step_out['recovery_outputs'].cpu().tolist() # [batch, seq, 2]
        # generate decision mask and lenghts
        if model_type == 'bert_char': # if char-level model
            mask = batch['input_decision_mask']
            lens = batch['input_mask'].sum(dim=-1).long()
        else:
            mask = batch['input_decision_mask']
            lens = batch['input_wordmask'].sum(dim=-1).long()
        # update counts for calculating F1
        B = list(lens.size())[0]
        for i in range(B):
            char2word = ori_batch['char2word'][i]
            decisions = {}
            for j in range(1, lens[i]): # [CLS] A B C ... [SEP]
                if mask[i,j] == 0.0:
                    continue
                c = recovery_out[i][j]
                if c == 0:
                    continue
                if model_type == 'bert_char':
                    j = char2word[j]
                decisions[j] = pro_mapping[c]
            recovery_decisions.append(decisions)
    return recovery_decisions


api.add_resource(ZPRecover, '/')

FLAGS = None
model = None
tokenizer = None
pro_mapping = None
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--prefix_path', type=str, required=True, help='Prefix path to the saved model')
    parser.add_argument('--in_path', type=str, required=True, help='Path to the input file.')
    parser.add_argument('--out_path', type=str, default=None, help='Path to the output file.')
    args, unparsed = parser.parse_known_args()
    FLAGS = config_utils.load_config(args.prefix_path + ".config.json")

    if FLAGS.model_type == 'bert_word':
        import zp_datastream
        import zp_model
    elif FLAGS.model_type == 'bert_char':
        import zp_datastream_char as zp_datastream
        import zp_model_char as zp_model
    else:
        assert False, "model_type '{}' not supported".format(FLAGS.model_type)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('device: {}, n_gpu: {}, grad_accum_steps: {}'.format(device, n_gpu, FLAGS.grad_accum_steps))

    tokenizer = None
    if 'bert' in FLAGS.pretrained_path:
        tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained_path)

    pro_mapping = json.load(open(FLAGS.pro_mapping, 'r'))
    print('Number of predefined pronouns: {}, they are: {}'.format(len(pro_mapping), pro_mapping.values()))

    print('Compiling model')
    model = zp_model.BertZP.from_pretrained(FLAGS.pretrained_path, char2word=FLAGS.char2word,
            pro_num=len(pro_mapping), max_relative_position=FLAGS.max_relative_position)
    model.load_state_dict(torch.load(args.prefix_path + ".bert_model.bin"))
    model.to(device)

    app.run(host='0.0.0.0', port=6060)

