
import os, sys, json, codecs
import numpy as np
import torch


def load_and_extract_features(path, tokenizer, data_type="recovery", char2word_strategy="last"):
    print('Data type: {}, char2word_strategy: {}'.format(data_type, char2word_strategy))
    data = json.load(open(path, 'r'))

    features = []
    sent_id_mapping = {}
    right, total = 0.0, 0.0
    for i, (sent_bert_toks, sent_bert_idxs) in enumerate(zip(data['sentences_bert_toks'], data['sentences_bert_idxs'])):
        if len(sent_bert_toks) > 512:
            print('Sentence No. {} length {}.'.format(i, len(sent_bert_toks)))
            continue
        sent_id_mapping[i] = len(features)
        sent_bert_toks = [x if x in tokenizer.vocab else '[UNK]' for x in sent_bert_toks]
        right += sum([x == '[UNK]' for x in sent_bert_toks])
        total += len(sent_bert_toks)
        input_ids = tokenizer.convert_tokens_to_ids(sent_bert_toks) # [seq]
        input_char2word = [] # [wordseq, wordlen OR 1]
        for idxs in sent_bert_idxs:
            if char2word_strategy == 'first':
                input_char2word.append(idxs[:1])
            elif char2word_strategy == 'last':
                input_char2word.append(idxs[-1:])
            elif char2word_strategy in ('mean', 'sum', ):
                input_char2word.append(idxs)
            else:
                assert False, 'Unsupported char2word_strategy: ' + char2word_strategy
        features.append({'input_ids':input_ids, 'input_char2word':input_char2word})
    print('OOV rate: {}, {}/{}'.format(right/total, right, total))

    if data_type == 'recovery':
        extract_recovery(data, features, sent_id_mapping)

    return features


def extract_resolution(data, features):
    pass


def extract_recovery(data, features, sent_id_mapping):
    for feat in features:
        input_char2word = feat['input_char2word']
        feat['input_zp'] = [0 for _ in range(len(input_char2word))] # [wordseq]
        feat['input_zp_cid'] = [0 for _ in range(len(input_char2word))] # [wordseq]

    for zp_inst in data['zp_info']:
        i, j = zp_inst['zp_sent_index'], zp_inst['zp_index']
        assert j >= 1 # There shouldn't be ZP before [CLS]
        if i not in sent_id_mapping:
            continue
        i = sent_id_mapping[i]
        pro_cid = zp_inst['recovery']
        assert type(pro_cid) == int
        features[i]['input_zp'][j] = 1
        features[i]['input_zp_cid'][j] = pro_cid


# (input_ids, input_char2word, input_zp, input_zp_cid)
def make_recovery_batch(features, batch_size, is_sort=True, is_random=False):
    if is_sort:
        features.sort(key=lambda x: len(x['input_ids']))
    elif is_random:
        random.shuffle(features)
    N = 0
    batches = []
    while N < len(features):
        B = min(batch_size, len(features)-N)
        maxseq, maxwordseq, maxwordlen = 0, 0, 0
        for i in range(0, B):
            maxseq = max(maxseq, len(features[N+i]['input_ids']))
            maxwordseq = max(maxwordseq, len(features[N+i]['input_char2word']))
            for x in features[N+i]['input_char2word']:
                maxwordlen = max(maxwordlen, len(x))
        input_ids = np.zeros([B, maxseq], dtype=np.long)
        input_mask = np.zeros([B, maxseq], dtype=np.float)
        input_wordmask = np.zeros([B, maxwordseq], dtype=np.float)
        input_char2word = np.zeros([B, maxwordseq, maxwordlen], dtype=np.long)
        input_char2word_mask = np.zeros([B, maxwordseq, maxwordlen], dtype=np.float)
        input_zp = np.zeros([B, maxwordseq], dtype=np.long)
        input_zp_cid = np.zeros([B, maxwordseq], dtype=np.long)
        for i in range(0, B):
            curseq = len(features[N+i]['input_ids'])
            curwordseq = len(features[N+i]['input_char2word'])
            input_ids[i,:curseq] = features[N+i]['input_ids']
            input_mask[i,:curseq] = [1,]*curseq
            input_wordmask[i,:curwordseq] = [1,]*curwordseq
            for j in range(0, curwordseq):
                curwordlen = len(features[N+i]['input_char2word'][j])
                input_char2word[i,j,:curwordlen] = features[N+i]['input_char2word'][j]
                input_char2word_mask[i,j,:curwordlen] = [1,]*curwordlen
            input_zp[i,:curwordseq] = features[N+i]['input_zp']
            input_zp_cid[i,:curwordseq] = features[N+i]['input_zp_cid']
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.float)
        input_wordmask = torch.tensor(input_wordmask, dtype=torch.float)
        input_char2word = torch.tensor(input_char2word, dtype=torch.long)
        input_char2word_mask = torch.tensor(input_char2word_mask, dtype=torch.float)
        input_zp = torch.tensor(input_zp, dtype=torch.long)
        input_zp_cid = torch.tensor(input_zp_cid, dtype=torch.long)


        batches.append({'input_ids':input_ids, 'input_mask':input_mask, 'input_wordmask':input_wordmask,
            'input_char2word':input_char2word, 'input_char2word_mask':input_char2word_mask,
            'input_zp':input_zp, 'input_zp_cid':input_zp_cid, 'input_zp_span':None, 'type':'recovery'})
        N += B
    return batches

