
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
        sent_bert_toks = [x if x in tokenizer.vocab else '[UNK]' for x in sent_bert_toks]
        right += sum([x == '[UNK]' for x in sent_bert_toks])
        total += len(sent_bert_toks)
        # Example sent_bert_idxs: [0] [1,2] [3] [4]; input_ids: [CLS] A B C [SEP]
        # input_decision_mask = [1, 1, 0, 1, 1]
        # word_id_mapping = {0:0, 1:1, 2:3, 3:4}
        word_id_mapping = {}
        input_ids = tokenizer.convert_tokens_to_ids(sent_bert_toks) # [seq]
        input_decision_mask = []
        for j, idxs in enumerate(sent_bert_idxs):
            curlen = len(input_decision_mask)
            word_id_mapping[j] = curlen
            input_decision_mask.extend([0 for _ in idxs])
            input_decision_mask[curlen] = 1
        assert all([input_decision_mask[v] for v in word_id_mapping.values()])
        assert len(input_ids) == len(input_decision_mask)
        features.append({'input_ids':input_ids, 'input_decision_mask':input_decision_mask})
        sent_id_mapping[i] = len(features)-1
    print('OOV rate: {}, {}/{}'.format(right/total, right, total))

    if data_type == 'recovery':
        extract_recovery(data, features, sent_id_mapping, word_id_mapping)

    return features


def extract_resolution(data, features):
    pass


def extract_recovery(data, features, sent_id_mapping, word_id_mapping):
    for feat in features:
        input_ids = feat['input_ids']
        feat['input_zp'] = [0 for _ in input_ids] # [seq]
        feat['input_zp_cid'] = [0 for _ in input_ids] # [seq]

    for zp_inst in data['zp_info']:
        i, j = zp_inst['zp_sent_index'], zp_inst['zp_index']
        assert j >= 1 # There shouldn't be ZP before [CLS]
        if i not in sent_id_mapping:
            continue
        i, j = sent_id_mapping[i], word_id_mapping[j]
        pro_cid = zp_inst['recovery']
        assert type(pro_cid) == int
        features[i]['input_zp'][j] = 1
        features[i]['input_zp_cid'][j] = pro_cid


# (input_ids, input_char2word, input_zp, input_zp_cid)
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
        input_zp = np.zeros([B, maxwordseq], dtype=np.long)
        input_zp_cid = np.zeros([B, maxwordseq], dtype=np.long)
        for i in range(0, B):
            curseq = len(features[N+i]['input_ids'])
            input_ids[i,:curseq] = features[N+i]['input_ids']
            input_mask[i,:curseq] = [1,]*curseq
            input_decision_mask[i,:curseq] = features[N+i][['input_decision_mask']
            input_zp[i,:curseq] = features[N+i]['input_zp']
            input_zp_cid[i,:curseq] = features[N+i]['input_zp_cid']
        input_ids = torch.tensor(input_ids, dtype=torch.long)
        input_mask = torch.tensor(input_mask, dtype=torch.float)
        input_decision_mask = torch.tensor(input_decision_mask, dtype=torch.float)
        input_zp = torch.tensor(input_zp, dtype=torch.long)
        input_zp_cid = torch.tensor(input_zp_cid, dtype=torch.long)


        batches.append({'input_ids':input_ids, 'input_mask':input_mask, 'input_decision_mask':input_decision_mask,
            'input_zp':input_zp, 'input_zp_cid':input_zp_cid, 'input_zp_span':None, 'type':'recovery'})
        N += B
    return batches


if __name__ == '__main__':
    pass
