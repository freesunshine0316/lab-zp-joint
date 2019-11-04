
import os, sys, json, codecs
import numpy as np
import torch


def load_and_extract_features(path, tokenizer, char2word="sum", data_type="recovery",
        is_gold_tree=True, is_only_azp=False):
    assert data_type in ("recovery", "resolution")
    print('Data type: {}, char2word: {}'.format(data_type, char2word))
    print("zp_datastream_char.py: for model_type 'bert_char', 'char2word' not in use")
    data = json.load(open(path, 'r'))

    if data_type == 'resolution':
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
        # Example: sent_bert_idxs: [0] [1, 2, 3] [4]; sent_bert_toks: [CLS] A B C [SEP]; decision_start: 1
        # input_decision_mask = [0, 1, 0, 1, 1]
        # input_word_boundary_mask = [1, 1, 0, 1, 1]
        decision_start = data['sentences_decision_start'][i] if 'sentences_decision_start' in data else 0
        input_decision_mask = []
        input_word_boundary_mask = []
        for j, idxs in enumerate(sent_bert_idxs):
            curlen = len(input_decision_mask)
            input_decision_mask.extend([0 for _ in idxs])
            if j >= decision_start:
                input_decision_mask[curlen] = 1
            input_word_boundary_mask.extend([0 for _ in idxs])
            input_word_boundary_mask[curlen] = 1
            input_word_boundary_mask[-1] = 1
        assert len(input_ids) == len(input_decision_mask) == len(input_word_boundary_mask)
        features.append({'input_ids':input_ids,
            'input_decision_mask':input_decision_mask, 'input_word_boundary_mask':input_word_boundary_mask})
        sent_id_mapping[i] = len(features) - 1
    print('OOV rate: {}, {}/{}'.format(right/total, right, total))

    if data_type == 'recovery':
        extract_recovery(data, features, sent_id_mapping)
    else:
        extract_resolution(data, features, sent_id_mapping,
                is_gold_tree=is_gold_tree, is_only_azp=is_only_azp)

    return features


def extract_resolution(data, features, sent_id_mapping, is_gold_tree=True, is_only_azp=False):
    for i in range(len(features)):
        input_ids = features[i]['input_ids']
        features[i]['input_zp'] = [0 for _ in input_ids] # [seq]
        features[i]['input_zp_span'] = [[(0,0),] for _ in input_ids] # [seq, list of span]

    for i, sent_nps in enumerate(data['sentences_nps']):
        if i not in sent_id_mapping:
            continue
        i = sent_id_mapping[i]
        features[i]['input_nps'] = set(tuple(x['span_char']) for x in sent_nps)
        features[i]['input_nps'].add((0,0)) # add (0,0) NP as None category
        #for st_char, ed_char in features[i]['input_nps']:
        #    assert features[i]['input_word_boundary_mask'][st_char] == 1
        #    assert features[i]['input_word_boundary_mask'][ed_char] == 1

    for zp_inst in data['zp_info']:
        i, j_char = zp_inst['zp_sent_index'], zp_inst['zp_char_index']
        assert j_char >= 1 # There shouldn't be ZP before [CLS]
        if i not in sent_id_mapping:
            continue
        i = sent_id_mapping[i]
        features[i]['input_zp'][j_char] = 1
        for k, (st_char, ed_char) in enumerate(zp_inst['resolution_char']):
            assert (st_char,ed_char) != (0,0) # span can't be (0,0), which represents 'None' for resolution
            assert ed_char < j_char # Resolution span should be less than ZP-index
            if is_gold_tree: # if gold tree, then span should be an NP
                assert (st_char,ed_char) in features[i]['input_nps']
            #assert features[i]['input_word_boundary_mask'][st_char] == 1
            #assert features[i]['input_word_boundary_mask'][ed_char] == 1
            if features[i]['input_zp_span'][j_char][-1] == (0,0):
                features[i]['input_zp_span'][j_char].pop()
            features[i]['input_zp_span'][j_char].append((st_char,ed_char))

    if is_only_azp:
        for i in range(len(features)):
            for j_char in range(len(features[i]['input_decision_mask'])):
                if features[i]['input_decision_mask'][j_char] > 0 and features[i]['input_zp'][j_char] == 0:
                    features[i]['input_decision_mask'][j_char] = 0

    ## check the AZP/ZP percentage, for each instance, only check the last sentence
    #right, total = 0.0, 0.0
    #for i in range(len(features)):
    #    for j_char in range(len(features[i]['input_zp_span'])):
    #        if feature[i]['input_decision_mask'][j_char] and features[i]['input_zp'][j_char]:
    #            total += 1.0
    #            right += (0,0) not in features[i]['input_zp_span'][j_char]
    #print('ana zp/zp: {}, {}, {}'.format(right/total, right, total))


def extract_recovery(data, features, sent_id_mapping):
    for inst in features:
        input_ids = inst['input_ids']
        inst['input_zp'] = [0 for _ in input_ids] # [seq]
        inst['input_zp_cid'] = [0 for _ in input_ids] # [seq]

    for zp_inst in data['zp_info']:
        i, j_char = zp_inst['zp_sent_index'], zp_inst['zp_char_index']
        assert j_char >= 1 # There shouldn't be ZP before [CLS]
        if i not in sent_id_mapping:
            continue
        i = sent_id_mapping[i]
        pro_cid = zp_inst['recovery']
        assert type(pro_cid) == int
        features[i]['input_zp'][j_char] = 1
        features[i]['input_zp_cid'][j_char] = pro_cid


def make_batch(data_type, features, batch_size, is_sort=True, is_shuffle=False):
    assert data_type in ("recovery", "resolution")
    if data_type == "recovery":
        return make_recovery_batch(features, batch_size, is_sort=is_sort, is_shuffle=is_shuffle)
    else:
        return make_resolution_batch(features, batch_size, is_sort=is_sort, is_shuffle=is_shuffle)


def make_resolution_batch(features, batch_size, is_sort=True, is_shuffle=False):
    if is_sort:
        features.sort(key=lambda x: len(x['input_ids']))
    elif is_shuffle:
        random.shuffle(features)
    N = 0
    batches = []
    while N < len(features):
        B = min(batch_size, len(features)-N)
        maxseq = 0
        for i in range(0, B):
            maxseq = max(maxseq, len(features[N+i]['input_ids']))
        input_ids = np.zeros([B, maxseq], dtype=np.long)
        input_mask = np.zeros([B, maxseq], dtype=np.float)
        input_decision_mask = np.zeros([B, maxseq], dtype=np.float)
        input_word_boundary_mask = np.zeros([B, maxseq], dtype=np.float)
        input_zp = np.zeros([B, maxseq], dtype=np.long)
        input_zp_span = np.zeros([B, maxseq, maxseq, 2], dtype=np.float)
        input_zp_span_multiref = [[] for i in range(0, B)] # [batch, seq, SET of spans]
        input_nps = [features[N+i]['input_nps'] for i in range(0, B)] # [batch, SET of spans]
        for i in range(0, B):
            curseq = len(features[N+i]['input_ids'])
            input_ids[i,:curseq] = features[N+i]['input_ids']
            input_mask[i,:curseq] = [1,]*curseq
            input_decision_mask[i,:curseq] = features[N+i]['input_decision_mask']
            input_word_boundary_mask[i,:curseq] = features[N+i]['input_word_boundary_mask']
            input_zp[i,:curseq] = features[N+i]['input_zp']
            input_zp_span_multiref[i] = [set() for j in range(0, curseq)]
            for j in range(0, curseq):
                span_num = len(features[N+i]['input_zp_span'][j])
                for st_char, ed_char in features[N+i]['input_zp_span'][j]:
                    input_zp_span[i,j,st_char,0] = 1.0/span_num
                    input_zp_span[i,j,ed_char,1] = 1.0/span_num
                    input_zp_span_multiref[i][j].add((st_char,ed_char))
                    if (st_char,ed_char) != (0,0):
                        assert ed_char < j
                if (0,0) in input_zp_span_multiref[i][j]:
                    assert len(input_zp_span_multiref[i][j]) == 1
                #assert (input_zp_span_multiref[i][j] & input_nps[i]) == input_zp_span_multiref[i][j]
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.float)
        input_decision_mask = torch.tensor(input_decision_mask, dtype=torch.float)
        input_word_boundary_mask = torch.tensor(input_word_boundary_mask, dtype=torch.float)
        input_zp = torch.tensor(input_zp, dtype=torch.long)
        input_zp_span = torch.tensor(input_zp_span, dtype=torch.float)


        batches.append({'input_ids':input_ids, 'input_mask':input_mask,
            'input_decision_mask':input_decision_mask, 'input_word_boundary_mask': input_word_boundary_mask,
            'input_zp':input_zp, 'input_zp_cid':None, 'input_zp_span':input_zp_span,
            'input_nps':input_nps, 'type':'resolution',
            'input_zp_span_multiref': input_zp_span_multiref})
        N += B
    return batches


def make_recovery_batch(features, batch_size, is_sort=True, is_shuffle=False):
    if is_sort:
        features.sort(key=lambda x: len(x['input_ids']))
    elif is_shuffle:
        random.shuffle(features)
    N = 0
    batches = []
    while N < len(features):
        B = min(batch_size, len(features)-N)
        maxseq = 0
        for i in range(0, B):
            maxseq = max(maxseq, len(features[N+i]['input_ids']))
        input_ids = np.zeros([B, maxseq], dtype=np.long)
        input_mask = np.zeros([B, maxseq], dtype=np.float)
        input_decision_mask = np.zeros([B, maxseq], dtype=np.float)
        input_word_boundary_mask = np.zeros([B, maxseq], dtype=np.float)
        input_zp = np.zeros([B, maxseq], dtype=np.long)
        input_zp_cid = np.zeros([B, maxseq], dtype=np.long)
        for i in range(0, B):
            curseq = len(features[N+i]['input_ids'])
            input_ids[i,:curseq] = features[N+i]['input_ids']
            input_mask[i,:curseq] = [1,]*curseq
            input_decision_mask[i,:curseq] = features[N+i]['input_decision_mask']
            input_word_boundary_mask[i,:curseq] = features[N+i]['input_word_boundary_mask']
            input_zp[i,:curseq] = features[N+i]['input_zp']
            input_zp_cid[i,:curseq] = features[N+i]['input_zp_cid']
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.float)
        input_decision_mask = torch.tensor(input_decision_mask, dtype=torch.float)
        input_word_boundary_mask = torch.tensor(input_word_boundary_mask, dtype=torch.float)
        input_zp = torch.tensor(input_zp, dtype=torch.long)
        input_zp_cid = torch.tensor(input_zp_cid, dtype=torch.long)


        batches.append({'input_ids':input_ids, 'input_mask':input_mask,
            'input_decision_mask':input_decision_mask, 'input_word_boundary_mask': input_word_boundary_mask,
            'input_zp':input_zp, 'input_zp_cid':input_zp_cid, 'input_zp_span':None,
            'type':'recovery'})
        N += B
    return batches


if __name__ == '__main__':
    pass
