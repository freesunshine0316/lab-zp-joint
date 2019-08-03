
import os, sys, json

from stanfordnlp.server import CoreNLPClient


class PseudoRoot:
    def __init__(self):
        self.value = "TOP"
        self.child = []

def recur_gen_str(root, level, is_indent):
    if len(root.child) == 0:
        return root.value
    string = "    "*level if is_indent else ""
    string += "("+root.value+" "
    string_child = "\n".join(recur_gen_str(child, level+1, i!=0) for i, child in enumerate(root.child))
    return string + string_child + ")"


def get_parse(sent, parser):
    anno = parser.annotate(sent)
    root = PseudoRoot()
    for sent in anno.sentence:
        assert len(sent.parseTree.child) == 1
        root.child.append(sent.parseTree.child[0])
    return recur_gen_str(root, 1, True)


def process_file(fin, fout, parser):
    is_intree, is_insent = False, False
    cur_sent = []
    lines = []
    for line in fin:
        line = line.rstrip()
        lines.append(line)

        if line == 'Treebanked sentence:':
            is_insent = True
        elif is_insent and line == "":
            is_insent = False
        elif is_insent and line.startswith('------------') == False:
            cur_sent.append(line.strip())

        if line == "Tree:":
            is_intree = True
            fout.write('Tree:\n-----\n')
            parse_tree = get_parse(' '.join(cur_sent), parser)
            fout.write(parse_tree+'\n\n')
            cur_sent = []
        elif is_intree and line == "":
            is_intree = False
        elif not is_intree:
            fout.write(line+'\n')


######################################################


properties={"tokenize.language": "zh",
            "tokenize.whitespace": "true",
            "ssplit.eolonly": "true",
            "segment.model": "edu/stanford/nlp/models/segmenter/chinese/ctb.gz",
            "segment.sighanCorporaDict": "edu/stanford/nlp/models/segmenter/chinese",
            "segment.serDictionary": "edu/stanford/nlp/models/segmenter/chinese/dict-chris6.ser.gz",
            "segment.sighanPostProcessing": "true",
            "ssplit.boundaryTokenRegex": "[.。]|[!?！？]+",
            "pos.model": "edu/stanford/nlp/models/pos-tagger/chinese-distsim/chinese-distsim.tagger",
            "ner.language": "chinese",
            "ner.model": "edu/stanford/nlp/models/ner/chinese.misc.distsim.crf.ser.gz",
            "ner.applyNumericClassifiers": "true",
            "ner.useSUTime": "false",
            "ner.fine.regexner.mapping": "edu/stanford/nlp/models/kbp/chinese/gazetteers/cn_regexner_mapping.tab",
            "ner.fine.regexner.noDefaultOverwriteLabels": "CITY,COUNTRY,STATE_OR_PROVINCE",
            "parse.model": "edu/stanford/nlp/models/srparser/chineseSR.ser.gz",
            "coref.algorithm": "neural",
            "coref.language": "chinese",
            "coref.md.liberalMD": "true",
}
annotators=['tokenize', 'ssplit', 'pos', 'lemma', 'ner', 'parse']

os.environ['CORENLP_HOME'] = '/data/home/lfsong/ws/exp.dialogue_zp/data.OntoNotes50/data/stanford-corenlp-full-2018-10-05'
with CoreNLPClient(properties=properties, annotators=annotators,
         endpoint="http://localhost:9091",
         timeout=60000, memory='16G', threads=8, output_format='serialized', be_quiet=False) as parser:
    for inpath in open('zp_data_fof', 'r'):
        inpath = inpath.strip()
        outpath = inpath.replace('.onf', '.onf_autotree')
        #if inpath.endswith("cnn_0001.onf") == False:
        #    continue
        if inpath.endswith('.onf') == False:
            continue
        fin = open(inpath, 'r')
        fout = open(outpath, 'w')
        print('processing {}'.format(inpath))
        process_file(fin, fout, parser)
        fin.close()
        fout.close()

