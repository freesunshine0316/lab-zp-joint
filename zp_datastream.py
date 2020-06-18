
import os, sys, json, codecs
import numpy as np
import torch


def load_and_extract_features(path, tokenizer, char2word="sum", data_type="recovery", is_only_azp=False):
    data = json.load(open(path, 'r'))
    return extract_features(data, tokenizer, char2word=char2word, data_type=data_type, is_only_azp=is_only_azp)


def extract_features(data, tokenizer, char2word="sum", data_type="recovery", is_only_azp=False):
    assert data_type.startswith("recovery") or data_type.startswith("resolution")
    print('Data type: {}, char2word: {}'.format(data_type, char2word))

    if data_type.startswith('resolution'):
        assert len(data['sentences_nps']) == len(data['sentences_bert_toks'])

    features = []
    sent_id_mapping = {}
    right, total = 0.0, 0.0
    for i, (sent_bert_toks, sent_bert_idxs) in enumerate(zip(data['sentences_bert_toks'], data['sentences_bert_idxs'])):
        if len(sent_bert_toks) > 512:
            print('Sentence No. {} length {}.'.format(i, len(sent_bert_toks)))
            continue
        sent_bert_toks = [x if x in tokenizer.vocab else '[UNK]' for x in sent_bert_toks]
        right += sum([x == '[UNK]' for x in sent_bert_toks])
        total += len(sent_bert_toks)
        input_ids = tokenizer.convert_tokens_to_ids(sent_bert_toks) # [seq]
        # Example: sent_bert_idxs: [0] [1, 2, 3] [4]; sent_bert_toks: [CLS] [A B C] [SEP]; decision_start: 1
        # input_decision_mask = [0, 1, 1]
        decision_start = data['sentences_decision_start'][i] if 'sentences_decision_start' in data else 0
        input_decision_mask = []
        input_char2word = [] # [wordseq, wordlen OR 1]
        char2word_map = {}
        for j, idxs in enumerate(sent_bert_idxs):
            input_decision_mask.append(1 if j >= decision_start else 0)
            if char2word == 'first':
                input_char2word.append(idxs[:1])
            elif char2word == 'last':
                input_char2word.append(idxs[-1:])
            elif char2word in ('mean', 'sum', ):
                input_char2word.append(idxs)
            else:
                assert False, 'Unsupported char2word: ' + char2word
            char2word_map.update({k:j for k in idxs})
        features.append({'input_ids':input_ids, 'input_char2word':input_char2word,
            'input_decision_mask':input_decision_mask, 'char2word':char2word_map})
        sent_id_mapping[i] = len(features) - 1
    print('OOV rate: {}, {}/{}'.format(right/total, right, total))

    is_inference = data_type.find('inference') >= 0
    if data_type.startswith('recovery'):
        extract_recovery(data, features, sent_id_mapping, is_inference=is_inference)
    elif data_type.startswith('resolution'):
        extract_resolution(data, features, sent_id_mapping, is_inference=is_inference, is_only_azp=is_only_azp)
    else:
        assert False, 'Unknown'

    return features


def extract_resolution(data, features, sent_id_mapping, is_inference=False, is_only_azp=False):
    for i, sent_nps in enumerate(data['sentences_nps']):
        if i not in sent_id_mapping:
            continue
        i = sent_id_mapping[i]
        features[i]['input_nps'] = set(tuple(x['span']) for x in sent_nps)
        features[i]['input_nps'].add((0,0)) # add (0,0) NP as None category

    if is_inference:
        return

    for i in range(len(features)):
        input_char2word = features[i]['input_char2word']
        features[i]['input_zp'] = [0 for _ in input_char2word] # [seq]
        features[i]['input_zp_span'] = [[(0,0),] for _ in input_char2word] # [seq, list of span]


    for zp_inst in data['zp_info']:
        i, j = zp_inst['zp_sent_index'], zp_inst['zp_index']
        assert j >= 1 # There shouldn't be ZP before [CLS]
        if i not in sent_id_mapping:
            continue
        i = sent_id_mapping[i]
        features[i]['input_zp'][j] = 1
        for k, (st,ed) in enumerate(zp_inst['resolution']):
            assert (st,ed) != (0,0) # Resolution span can't be (0,0), which represents 'None'
            assert ed < j # Resolution span should be less than ZP-index
            #if is_gold_tree: # if gold tree, then span should be an NP
            #    assert (st,ed) in features[i]['input_nps']
            if features[i]['input_zp_span'][j][-1] == (0,0):
                features[i]['input_zp_span'][j].pop()
            features[i]['input_zp_span'][j].append((st,ed))

    if is_only_azp:
        for i in range(len(features)):
            for j in range(len(features[i]['input_decision_mask'])):
                if features[i]['input_decision_mask'][j] > 0 and features[i]['input_zp'][j] == 0:
                    features[i]['input_decision_mask'][j] = 0

    ## check the AZP/ZP percentage, for each instance, only check the last sentence
    #right, total = 0.0, 0.0
    #for i in range(len(features)):
    #    for j in range(len(features[i]['input_zp_span'])):
    #        if feature[i]['input_decision_mask'][j] and features[i]['input_zp'][j]:
    #            total += 1.0
    #            right += (0,0) not in features[i]['input_zp_span'][j]
    #print('ana zp/zp: {}, {}, {}'.format(right/total, right, total))


def extract_recovery(data, features, sent_id_mapping, is_inference=False):
    if is_inference:
        return

    for feat in features:
        input_char2word = feat['input_char2word']
        feat['input_zp'] = [0 for _ in input_char2word] # [wordseq]
        feat['input_zp_cid'] = [0 for _ in input_char2word] # [wordseq]

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


def make_batch(data_type, features, batch_size, is_sort=True, is_shuffle=False):
    assert data_type.startswith("recovery") or data_type.startswith("resolution")
    is_inference = data_type.find('inference') >= 0
    if data_type.startswith("recovery"):
        return make_recovery_batch(features, batch_size,
                is_inference=is_inference, is_sort=is_sort, is_shuffle=is_shuffle)
    elif data_type.startswith("resolution"):
        return make_resolution_batch(features, batch_size,
                is_inference=is_inference, is_sort=is_sort, is_shuffle=is_shuffle)
    else:
        assert False, 'Unknown'


def make_resolution_batch(features, batch_size, is_inference=False, is_sort=True, is_shuffle=False):
    if is_sort:
        features.sort(key=lambda x: len(x['input_ids']))
    elif is_shuffle:
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
        input_char2word = np.zeros([B, maxwordseq, maxwordlen], dtype=np.long)
        input_char2word_mask = np.zeros([B, maxwordseq, maxwordlen], dtype=np.float)
        input_wordmask = np.zeros([B, maxwordseq], dtype=np.float)
        input_decision_mask = np.zeros([B, maxwordseq], dtype=np.float)
        input_zp = np.zeros([B, maxwordseq], dtype=np.long)
        input_zp_span = np.zeros([B, maxwordseq, maxwordseq, 2], dtype=np.float)
        input_zp_span_multiref = [[] for i in range(0, B)] # [batch, wordseq, SET of spans]
        input_nps = [features[N+i]['input_nps'] for i in range(0, B)] # [batch, SET of spans]
        for i in range(0, B):
            curseq = len(features[N+i]['input_ids'])
            curwordseq = len(features[N+i]['input_char2word'])
            input_ids[i,:curseq] = features[N+i]['input_ids']
            input_mask[i,:curseq] = [1,]*curseq
            for j in range(0, curwordseq):
                curwordlen = len(features[N+i]['input_char2word'][j])
                input_char2word[i,j,:curwordlen] = features[N+i]['input_char2word'][j]
                input_char2word_mask[i,j,:curwordlen] = [1,]*curwordlen
            input_wordmask[i,:curwordseq] = [1,]*curwordseq
            input_decision_mask[i,:curwordseq] = features[N+i]['input_decision_mask']
            input_zp[i,:curwordseq] = features[N+i]['input_zp']
            input_zp_span_multiref[i] = [set() for j in range(0, curwordseq)]
            for j in range(0, curwordseq):
                span_num = len(features[N+i]['input_zp_span'][j])
                for st, ed in features[N+i]['input_zp_span'][j]:
                    input_zp_span[i,j,st,0] = 1.0/span_num
                    input_zp_span[i,j,ed,1] = 1.0/span_num
                    input_zp_span_multiref[i][j].add((st,ed))
                    if (st,ed) != (0,0):
                        assert ed < j
                if (0,0) in input_zp_span_multiref[i][j]:
                    assert len(input_zp_span_multiref[i][j]) == 1
                #assert (input_zp_span_multiref[i][j] & input_nps[i]) == input_zp_span_multiref[i][j]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.float)
        input_char2word = torch.tensor(input_char2word, dtype=torch.long)
        input_char2word_mask = torch.tensor(input_char2word_mask, dtype=torch.float)
        input_wordmask = torch.tensor(input_wordmask, dtype=torch.float)
        input_decision_mask = torch.tensor(input_decision_mask, dtype=torch.float)
        input_word_boundary_mask = torch.tensor(input_word_boundary_mask, dtype=torch.float)
        input_zp = torch.tensor(input_zp, dtype=torch.long)
        input_zp_span = torch.tensor(input_zp_span, dtype=torch.float)


        batches.append({'input_ids':input_ids, 'input_mask':input_mask, 'input_nps':input_nps, 'type':'resolution',
            'input_char2word':input_char2word, 'input_char2word_mask':input_char2word_mask,
            'input_wordmask': input_wordmask, 'input_decision_mask':input_decision_mask,
            'input_zp':input_zp, 'input_zp_cid':None, 'input_zp_span':input_zp_span,
            'char2word':[features[N+i]['char2word'] for i in range(0, B)],
            'input_zp_span_multiref': input_zp_span_multiref})
        N += B
    return batches


def make_recovery_batch(features, batch_size, is_inference=False, is_sort=True, is_shuffle=False):
    if is_sort:
        features.sort(key=lambda x: len(x['input_ids']))
    elif is_shuffle:
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
        input_char2word = np.zeros([B, maxwordseq, maxwordlen], dtype=np.long)
        input_char2word_mask = np.zeros([B, maxwordseq, maxwordlen], dtype=np.float)
        input_wordmask = np.zeros([B, maxwordseq], dtype=np.float)
        input_decision_mask = np.zeros([B, maxwordseq], dtype=np.float)
        input_zp = np.zeros([B, maxwordseq], dtype=np.long)
        input_zp_cid = np.zeros([B, maxwordseq], dtype=np.long)
        for i in range(0, B):
            curseq = len(features[N+i]['input_ids'])
            curwordseq = len(features[N+i]['input_char2word'])
            input_ids[i,:curseq] = features[N+i]['input_ids']
            input_mask[i,:curseq] = [1,]*curseq
            for j in range(0, curwordseq):
                curwordlen = len(features[N+i]['input_char2word'][j])
                input_char2word[i,j,:curwordlen] = features[N+i]['input_char2word'][j]
                input_char2word_mask[i,j,:curwordlen] = [1,]*curwordlen
            input_wordmask[i,:curwordseq] = [1,]*curwordseq
            input_decision_mask[i,:curwordseq] = features[N+i]['input_decision_mask']
            input_zp[i,:curwordseq] = features[N+i]['input_zp']
            input_zp_cid[i,:curwordseq] = features[N+i]['input_zp_cid']
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.float)
        input_char2word = torch.tensor(input_char2word, dtype=torch.long)
        input_char2word_mask = torch.tensor(input_char2word_mask, dtype=torch.float)
        input_wordmask = torch.tensor(input_wordmask, dtype=torch.float)
        input_decision_mask = torch.tensor(input_decision_mask, dtype=torch.float)
        input_zp = torch.tensor(input_zp, dtype=torch.long)
        input_zp_cid = torch.tensor(input_zp_cid, dtype=torch.long)


        batches.append({'input_ids':input_ids, 'input_mask':input_mask,
            'input_char2word':input_char2word, 'input_char2word_mask':input_char2word_mask,
            'input_wordmask':input_wordmask, 'input_decision_mask':input_decision_mask,
            'input_zp':input_zp, 'input_zp_cid':input_zp_cid, 'input_zp_span':None,
            'char2word':[features[N+i]['char2word'] for i in range(0, B)],
            'type':'recovery'})
        N += B
    return batches

