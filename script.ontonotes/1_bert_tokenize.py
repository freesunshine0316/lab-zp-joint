
import os, sys, json, codecs
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer


def separate(data, split_every):
    data_train = {'sentences':[], 'sentences_bert_toks':[], 'sentences_bert_idxs':[], 'zp_info':[]}
    data_dev = {'sentences':[], 'sentences_bert_toks':[], 'sentences_bert_idxs':[], 'zp_info':[]}
    i_mapping = {}
    for i, (sent, bert_toks, bert_idxs) in enumerate(
            zip(data['sentences'], data['sentences_bert_toks'], data['sentences_bert_idxs'])):
        if (i+1) % split_every == 0: # 4 9 14 ...
            data_dev['sentences'].append(sent)
            data_dev['sentences_bert_toks'].append(bert_toks)
            data_dev['sentences_bert_idxs'].append(bert_idxs)
            i_mapping[i] = len(data_dev['sentences_bert_toks']) - 1
        else:
            data_train['sentences'].append(sent)
            data_train['sentences_bert_toks'].append(bert_toks)
            data_train['sentences_bert_idxs'].append(bert_idxs)
            i_mapping[i] = len(data_train['sentences_bert_toks']) - 1
    for zp_inst in data['zp_info']:
        i = zp_inst['zp_sent_index']
        new_zp_inst = zp_inst.copy()
        new_zp_inst['zp_sent_index'] = i_mapping[i]
        data = data_dev if (i+1) % split_every == 0 else data_train
        data['zp_info'].append(new_zp_inst)
    return data_train, data_dev


def get_char_idx(bert_idxs, i, j):
    return bert_idxs[i][j][0]


# add [CLS] and [SEP] tokens for each sentece
# update index information for zp_info
def process(path, tokenizer, split_every=0):
    data = json.load(open(path, 'r'))

    tok_total = 0.0
    data['sentences_bert_toks'] = [] # [batch, A1 A2 A3 B1 B2 C1 C2 ...]
    data['sentences_bert_idxs'] = [] # [batch, [0, 1, 2], [3, 4], [5, 6] ...]
    for i, sent in enumerate(data['sentences']):
        idxs = [[0],]
        toks = ['[CLS]',]
        for word in sent.split():
            idxs.append([])
            for char in tokenizer.tokenize(word):
                idxs[-1].append(len(toks))
                toks.append(char)
            if len(idxs[-1]) == 0:
                idxs.pop()
        idxs.append([len(toks)])
        toks.append('[SEP]')
        tok_total += len(toks)
        data['sentences_bert_toks'].append(toks)
        data['sentences_bert_idxs'].append(idxs)
    print('total number of tokens: {}'.format(tok_total))

    same, total = 0.0, 0.0
    new_zp_info = []
    for zp_inst in data['zp_info']:
        zp_inst['zp_index'] += 1
        i, j = zp_inst['zp_sent_index'], zp_inst['zp_index']
        j_char = get_char_idx(data['sentences_bert_idxs'], i, j)
        #print(data['sentences_bert_toks'][i])
        #print('{} ==> {}'.format(j, j_char))
        new_zp_inst = {'zp_sent_index':i, 'zp_index':j, 'zp_char_index':j_char,
                'resolution':[], 'resolution_char':[]}
        for ana_i, ana_st, ana_ed in zp_inst['ana_spans']:
            ana_st += 1
            ana_ed += 1
            if i-3 < ana_i < i or (ana_i == i and ana_ed < j):
                new_zp_inst['resolution'].append((ana_st,ana_ed))
                ana_char_st = get_char_idx(data['sentences_bert_idxs'], ana_i, ana_st)
                ana_char_ed = get_char_idx(data['sentences_bert_idxs'], ana_i, ana_ed+1)-1
                new_zp_inst['resolution_char'].append((ana_char_st,ana_char_ed))
                #print(data['sentences_bert_toks'][i][ana_char_st:ana_char_ed+1])
        new_zp_info.append(new_zp_inst)
        total += 1.0 if len(zp_inst['ana_spans']) > 0 else 0.0
        same += any([ana_span[0] in (i,i-1,i-2) for ana_span in zp_inst['ana_spans']])
    data['zp_info'] = new_zp_info
    print('same-sentence zp percent: {} = {} / {}'.format(same/total, same, total))

    if split_every > 0:
        assert split_every > 1
        data_train, data_dev = separate(data, split_every)
        json.dump(data_train, open(path+'_tokv2_train', 'w'))
        json.dump(data_dev, open(path+'_tokv2_dev', 'w'))
    else:
        json.dump(data, open(path+'_tokv2', 'w'))

################################

tokenizer = BasicTokenizer()
process('test_data.json', tokenizer)
process('train_data.json', tokenizer, split_every=5)


