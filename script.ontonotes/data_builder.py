#coding=utf8
import os
import sys
import re
import argparse
import math
import timeit
import numpy
import random
import codecs
from subprocess import *

from collections import defaultdict

from conf import *
from buildTree import get_info_from_file
import utils
from data_generater import *
random.seed(0)
numpy.random.seed(0)

import cPickle
sys.setrecursionlimit(1000000)

MAX = 2

def get_sentence(zp_sentence_index,zp_index,nodes_info):
    nl,wl = nodes_info[zp_sentence_index]
    return_words = []
    for i in range(len(wl)):
        this_word = wl[i].word
        if i == zp_index:
            return_words.append("**pro**")
        else:
            if not (this_word == "*pro*"):
                return_words.append(this_word)
    return " ".join(return_words)


def get_candi_info(candi_sentence_index,nodes_info,candi_begin,candi_end,res_result):
    nl,wl = nodes_info[candi_sentence_index]
    candi_word = []
    for i in range(candi_begin,candi_end+1):
        candi_word.append(wl[i].word)
    candi_word = "_".join(candi_word)

    candi_info = [str(res_result),candi_word]
    return candi_info


def setup():
    utils.mkdir(args.data)
    #utils.mkdir(args.data+"train/")
    #utils.mkdir(args.data+"test/")


def get_prev_index(zp_index, wi2realwi_map):
    while zp_index > 0:
        if zp_index-1 in wi2realwi_map:
            break
        zp_index = zp_index - 1
    if zp_index > 0: # A B *OP* *pro* C ==> 1
        return wi2realwi_map[zp_index-1]
    else: # *OP* *pro* A B C
        return -1


def is_zp(word):
    return len(word) > 2 and word.count('*') >= 2


def generate_data(files):
    paths = [w.strip() for w in open(files).readlines()]

    total_sentence_num = 0
    sentences = []
    sentences_ori = []
    noun_phrases = []
    zp_info = defaultdict(list)

    azp_in_np, azp_total = 0.0, 0.0
    zp_anaph, zp_total = 0.0, 0.0

    startt = timeit.default_timer()
    for p in paths:
        if p.strip().endswith("DS_Store"): continue
        file_name = p.strip()
        if file_name.endswith("onf"):
            #file_name += "_autotree"
            zps, azps, nps, nodes = get_info_from_file(file_name, 2)

            # generate mappings, store sentences
            senti2globalsenti = {} # sentence id mapping from local file to global
            wi2realwi = {} # for each k, word id mapping from with ZP to without ZP
            for k in nodes.keys():
                senti2globalsenti[k] = total_sentence_num
                total_sentence_num += 1
                nl, wl = nodes[k]
                wi2realwi[total_sentence_num-1] = {}
                realwl = []
                i2 = 0
                for i1, w in enumerate(wl):
                    w = w.word
                    if is_zp(w) == False:
                        wi2realwi[total_sentence_num-1][i1] = i2
                        i2 += 1
                        realwl.append(w)
                sentences.append(realwl)
                sentences_ori.append([w.word for w in wl])

            # generate NP information
            for k in nps.keys():
                nps_new = []
                cur_sentence_num = senti2globalsenti[k]
                # A B *pro* [ *OP* C D *pro* ] *OP* E F
                for (st_index, ed_index) in nps[k]:
                    #print ' '.join(sentences_ori[cur_sentence_num][st_index:ed_index+1]).decode('utf-8')
                    st = get_prev_index(st_index, wi2realwi[cur_sentence_num])+1
                    ed = get_prev_index(ed_index+1, wi2realwi[cur_sentence_num])
                    #print ' '.join(sentences[cur_sentence_num][st:ed+1]).decode('utf-8')
                    #print '====='
                    nps_new.append((st,ed))
                noun_phrases.append(nps_new)

            # generate zp information
            zp2ana = {} # (zp-sent, zp) ==> list of (candi-sent, candi-begin, candi-end)
            for (zp_sent_index, zp_index, antecedents, coref_id) in azps:
                zp_sent_index = senti2globalsenti[zp_sent_index]
                zp_index = get_prev_index(zp_index, wi2realwi[zp_sent_index])+1
                #A = ' '.join(sentences[zp_sent_index][:zp_index])
                #B = ' '.join(sentences[zp_sent_index][zp_index:])
                #print (A + ' *pro* ' + B).decode('utf-8')
                is_match = not len(antecedents) # if no antecedents, then we consider it matched
                zp2ana[(zp_sent_index, zp_index)] = []
                for (candi_sent_index, candi_begin_index, candi_end_index, coref_id) in antecedents:
                    candi_sent_index = senti2globalsenti[candi_sent_index]
                    #print ' '.join(sentences_ori[candi_sent_index][candi_begin_index:candi_end_index+1]).decode('utf8')
                    candi_begin_index = get_prev_index(candi_begin_index, wi2realwi[candi_sent_index])+1
                    candi_end_index = get_prev_index(candi_end_index+1, wi2realwi[candi_sent_index])
                    #print ' '.join(sentences[candi_sent_index][candi_begin_index:candi_end_index+1]).decode('utf8')
                    #print '====='
                    # previous two sentences, or same but before zp_index
                    if zp_sent_index - 3 < candi_sent_index < zp_sent_index or \
                            (candi_sent_index == zp_sent_index and candi_end_index < zp_index):
                        is_match |= (candi_begin_index,candi_end_index) in noun_phrases[candi_sent_index]
                        zp2ana[(zp_sent_index, zp_index)].append((candi_sent_index,
                            candi_begin_index, candi_end_index))
                azp_in_np += is_match
                azp_total += 1.0

            for (zp_sent_index, zp_index) in zps:
                zp_sent_index = senti2globalsenti[zp_sent_index]
                zp_index = get_prev_index(zp_index, wi2realwi[zp_sent_index])+1
                if (zp_sent_index, zp_index) not in zp2ana:
                    zp2ana[(zp_sent_index, zp_index)] = []

            for k,v in zp2ana.items():
                zp_total += 1.0
                zp_anaph += len(v) > 0

            # store zp information
            for k, v in zp2ana.items():
                zp_sent_index, zp_index = k
                v = sorted(v)
                zp_info[zp_sent_index].append({'zp_index':zp_index, 'ana_spans':v})
    print('AZP percent in NP: {}, {}, {}'.format(azp_in_np/azp_total, azp_in_np, azp_total))
    print('Anaphora percent in ZPs: {}, {}, {} '.format(zp_anaph/zp_total, zp_anaph, zp_total))

    for i in range(len(sentences)):
        sentences[i] = (' '.join(sentences[i])).decode('utf-8')
        sentences_ori[i] = (' '.join(sentences_ori[i])).decode('utf-8')

    endt = timeit.default_timer()
    print >> sys.stderr
    print >> sys.stderr, "Total use %.3f seconds for Data Generating"%(endt-startt)
    return zp_info, sentences, noun_phrases


# for auto tree, append "onf" with "onf_autotree" at around L88
def generate_json_data(file_name=""):

    DATA = args.raw_data

    train_data_path = args.data + "train_data.json_v2"
    test_data_path = args.data + "test_data.json_v2"
    #test_data_path = args.data + "test_data.json_v2_autotree"

    train_zp_info, train_sentences, train_nps = generate_data("./data/train_list")
    json.dump({'sentences':train_sentences, 'zp_info':train_zp_info, 'nps':train_nps},
            codecs.open(train_data_path, 'w', 'utf-8'))

    test_zp_info, test_sentences, test_nps = generate_data("./data/test_list")
    json.dump({'sentences':test_sentences, 'zp_info':test_zp_info, 'nps':test_nps},
            codecs.open(test_data_path, 'w', 'utf-8'))

if __name__ == "__main__":
    # build data from raw OntoNotes data
    setup()
    generate_json_data()

