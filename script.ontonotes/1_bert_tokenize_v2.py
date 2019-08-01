
import os, sys, json, codecs
import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer


def separate(data, split_every):
    data_train = {'sentences_nps':[], 'sentences_bert_toks':[], 'sentences_bert_idxs':[], 'zp_info':[]}
    data_dev = {'sentences_nps':[], 'sentences_bert_toks':[], 'sentences_bert_idxs':[], 'zp_info':[]}
    i_mapping = {}
    for i, (sent_nps, bert_toks, bert_idxs) in enumerate(
            zip(data['sentences_nps'], data['sentences_bert_toks'], data['sentences_bert_idxs'])):
        if (i+1) % split_every == 0: # 4 9 14 ...
            data_dev['sentences_nps'].append(sent_nps)
            data_dev['sentences_bert_toks'].append(bert_toks)
            data_dev['sentences_bert_idxs'].append(bert_idxs)
            i_mapping[i] = len(data_dev['sentences_bert_toks']) - 1
        else:
            data_train['sentences_nps'].append(sent_nps)
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


def get_char_idx(bert_idxs, j, is_last_char):
    idx = -1 if is_last_char else 0
    return bert_idxs[j][idx]


# add [CLS] and [SEP] tokens for each sentece
# update index information for zp_info
def process(path, tokenizer, split_every=0, is_goldtree=True):
    data = json.load(open(path, 'r'))
    data['sentences_bert_toks'] = [] # [batch, A1 A2 A3 B1 B2 C1 C2 ...]
    data['sentences_bert_idxs'] = [] # [batch, [0, 1, 2], [3, 4], [5, 6] ...]
    data['sentences_nps'] = [] # [batch, list of NPs]
    data['sentences_decision_start'] = [] # [batch]
    data_zp_info = []
    for i in range(0, len(data['sentences'])): # each instance, right most point
        idxs = [[0],]
        toks = ['[CLS]',]
        nps = []
        nps_set = {}
        all_offsets = {}
        # window size: 3-sentence
        # handle the special cases of first and first two sentences
        for j in range(max(i-2,0), i+1):
            offset = len(idxs)
            all_offsets[j] = offset
            if j == i:
                data['sentences_decision_start'].append(offset)
            for word in data['sentences'][j].split():
                idxs.append([])
                for char in tokenizer.tokenize(word):
                    idxs[-1].append(len(toks))
                    toks.append(char)
                if len(idxs[-1]) == 0:
                    idxs.pop()
            for st, ed in data['nps'][j]:
                st, ed = st+offset, ed+offset
                st_char = get_char_idx(idxs, st, is_last_char=False)
                ed_char = get_char_idx(idxs, ed, is_last_char=True)
                nps.append({'span':[st,ed], 'span_char':[st_char,ed_char]})
                nps_set[(st,ed)] = len(nps)-1
            if str(j) not in data['zp_info']:
                data['zp_info'][str(j)] = []
            # only process the ZP info for the last sentence
            if j < i:
                continue
            for zp_inst in data['zp_info'][str(j)]:
                zp_index = zp_inst['zp_index']+offset
                zp_char_index = get_char_idx(idxs, zp_index, is_last_char=False)
                new_zp_inst = {'zp_sent_index':i, 'zp_index':zp_index, 'zp_char_index':zp_char_index,
                        'resolution':[], 'resolution_char':[]}
                for candi_sent_index, candi_st, candi_ed in zp_inst['ana_spans']:
                    # data_builder has the [-2,0] constraint,
                    # here only make sure candi and zp are in the same instance
                    assert max(i-2,0) <= candi_sent_index <= j
                    candi_offset = all_offsets[candi_sent_index]
                    candi_st, candi_ed = candi_st+candi_offset, candi_ed+candi_offset
                    assert (candi_st,candi_ed) != (0,0)
                    # If we have the gold tree and zp candidate is not an NP, just skip it
                    if is_goldtree and (candi_st,candi_ed) not in nps_set:
                        continue
                    assert candi_ed < zp_index
                    new_zp_inst['resolution'].append((candi_st,candi_ed))
                    candi_char_st = get_char_idx(idxs, candi_st, is_last_char=False)
                    candi_char_ed = get_char_idx(idxs, candi_ed, is_last_char=True)
                    assert candi_char_ed < zp_char_index
                    new_zp_inst['resolution_char'].append((candi_char_st,candi_char_ed))
                data_zp_info.append(new_zp_inst)
        idxs.append([len(toks)])
        toks.append('[SEP]')
        data['sentences_bert_toks'].append(toks)
        data['sentences_bert_idxs'].append(idxs)
        data['sentences_nps'].append(nps)
    del data['sentences']
    del data['nps']
    data['zp_info'] = data_zp_info
    assert len(data['sentences_nps']) == len(data['sentences_bert_toks']) == len(data['sentences_decision_start'])

    outpath = path.replace('_v2', '')
    if split_every > 0:
        assert split_every > 1
        data_train, data_dev = separate(data, split_every)
        json.dump(data_train, open(outpath+'_tok_train_v2', 'w'))
        json.dump(data_dev, open(outpath+'_tok_dev_v2', 'w'))
    else:
        json.dump(data, open(outpath+'_tok_v2', 'w'))

################################

tokenizer = BasicTokenizer()
#process('test_data.json_v2', tokenizer)
#process('train_data.json_v2', tokenizer, split_every=5)
process('test_data.json_v2_auto', tokenizer)


