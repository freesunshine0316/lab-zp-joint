
import os, sys, json
import argparse
import numpy as np
import time
import random

import torch
import torch.nn as nn

import config_utils

import benepar
from nltk.tree import Tree
from pytorch_pretrained_bert.tokenization import BertTokenizer

def extract_spans_recur(root, offset, chunks, labels):
    width = 0
    for child in root:
        if isinstance(child, Tree):
            sub_width = extract_spans_recur(child, offset+width, chunks, labels)
            width += sub_width
        else:
            width += 1

    if root.label() in labels:
        chunks.append((offset, offset+width))

    #assert width == len(root.leaves())
    return width


def preprocess(inpath, outpath, parser):
    fout = open(outpath, 'w')
    for i, line in enumerate(open(inpath, 'r')):
        if i and i%100 == 0:
            print('{} '.format(i), end=' ')
        instance = json.loads(line.strip())
        trees = []
        for i, sentence in enumerate(instance['conversation']):
            sentence = sentence.strip().split()
            tree_str = ' '.join(str(parser.parse(sentence)).replace('\n',' ',100).split())
            trees.append(tree_str)
        jsonline = json.dumps({'conversation':instance['conversation'], 'trees':trees})
        fout.write(jsonline+'\n')
    fout.close()

parser = benepar.Parser("benepar_zh")
preprocess('dev.txt', 'dev.txt_zp', parser)
preprocess('train.txt', 'train.txt_zp', parser)

