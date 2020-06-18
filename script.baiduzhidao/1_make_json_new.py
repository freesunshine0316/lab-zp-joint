
import csv
import os, sys, json, codecs

import torch
from pytorch_pretrained_bert.tokenization import BertTokenizer, BasicTokenizer

def make_json(raw_data, tokenizer, pro_dict_rev):
    data = {'sentences': [], # [batch, wordseq]
            'sentences_bert_idxs': [], # [batch, wordseq, wordlen]
            'sentences_bert_toks': [], # [batch, seq]
            'zp_info': []} # [a sequence of ...]
    zpnum, total = 0.0, 0.0
    for i, (sentence, cids) in enumerate(raw_data):
        sent = ['[CLS]',] # [word]
        sent_bert_idxs = [[0],] # [word, char]
        sent_bert_toks = ['[CLS]',] # [char]
        sent_zp = [] # all zp in this sentence
        j = 1
        j_char = 1
        is_cut = False
        for word in sentence:
            if len(word) > 2 and word[0] == '*' and word[-1] == '*': # ZP
                cid = cids.pop(0)
                sent_zp.append({'zp_index':j, 'zp_char_index':j_char, 'zp_sent_index':i, 'recovery':cid})
                zpnum += 1.0
            else:
                sent.append(word)
                j += 1
                total += 1.0
                sent_bert_idxs.append([])
                for char in tokenizer.tokenize(word):
                    sent_bert_idxs[-1].append(len(sent_bert_toks))
                    sent_bert_toks.append(char)
                    j_char += 1
                if len(sent_bert_idxs[-1]) == 0:
                    sent.pop()
                    j -= 1
                    total -= 1.0
                    sent_bert_idxs.pop()
            #if len(sent_bert_toks) > 500:
            #    is_cut = True
            #    break
        if is_cut == False:
            assert len(cids) == 0
        sent.append('[SEP]')
        sent_bert_idxs.append([len(sent_bert_toks)])
        sent_bert_toks.append('[SEP]')
        data['sentences'].append(sent)
        data['sentences_bert_idxs'].append(sent_bert_idxs)
        data['sentences_bert_toks'].append(sent_bert_toks)
        data['zp_info'].extend(sent_zp)

    print('Num of sentences {}'.format(len(data['sentences'])))
    print('zp_appears / total: {}, {} / {}'.format(zpnum/total, zpnum, total))
    return data


#######################################


# generate zp mapping
if os.path.isfile('pro_dict_rev.json'):
    pro_dict_rev = json.load(open('pro_dict_rev.json','r'))
    pro_dict_rev = {int(k):v for k,v in pro_dict_rev.items()}
else:
    pro_dict_rev = {0:None,}
    with open('zhidao_denoise_1_new.csv') as csvfile:
        for i, row in enumerate(csv.reader(csvfile, delimiter='\t')):
            if len(row) > 3 and row[3].strip() != '':
                print(row[3])
                cid, pro = row[3].split(':')
                cid = int(cid)
                pro = pro.strip()
                assert cid not in pro_dict_rev
                pro_dict_rev[cid] = pro
    json.dump(pro_dict_rev, open('pro_dict_rev.json','w'))


print(pro_dict_rev)


# generate data
train, dev, test = [], [], []
with open('zhidao_denoise_1.csv') as csvfile:
    for i, row in enumerate(csv.reader(csvfile, delimiter=',')):
        if i == 0:
            continue
        id = int(row[0])
        sentence = row[1].strip().split()
        cids = json.loads(row[2])
        if 0 <= id and id <= 1037:
            train.append((sentence, cids))
        elif 1038 <= id and id <= 1253:
            dev.append((sentence, cids))
        elif 1254 <= id and id <= 1442:
            test.append((sentence, cids))
        else:
            assert False, 'shit... {}'.format(id)


tokenizer = BasicTokenizer()
json.dump(make_json(train, tokenizer, pro_dict_rev), open('train_new.json', 'w'))
json.dump(make_json(dev, tokenizer, pro_dict_rev), open('dev_new.json', 'w'))
json.dump(make_json(test, tokenizer, pro_dict_rev), open('test_new.json', 'w'))

