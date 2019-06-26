
import os, sys, json, codecs
import argparse
import numpy as np
import time
import random

import torch
import torch.nn as nn
from tqdm import tqdm, trange

from pytorch_pretrained_bert.tokenization import BertTokenizer
from pytorch_pretrained_bert.optimization import BertAdam, WarmupLinearSchedule

import config_utils
import zp_datastream, zp_model

FLAGS = None


def calc_seq_f1(outputs, refs):
    n_out, n_ref, n_both = 0.0, 0.0, 0.0
    for i in range(len(outputs)):
        for j in range(len(outputs[i])):
            if outputs[i][j] > 0.0:
                n_out += 1.0
            if refs[i][j] > 0.0:
                n_ref += 1.0
                if outputs[i][j] == refs[i][j]:
                    n_both += 1.0
    #print('{} {} {}'.format(n_both, n_out, n_ref))
    pr = n_both/n_out if n_out > 0.0 else 0.0
    rc = n_both/n_ref if n_ref > 0.0 else 0.0
    f1 = 2.0*pr*rc/(pr+rc) if pr > 0.0 and rc > 0.0 else 0.0
    return pr, rc, f1


def copy_batch(src, tgt, lens):
    for i, l in enumerate(lens):
        tgt.append(src[i][1:l-1])


def dev_eval(model, device, dev_batches):
    model.eval()
    print('Evaluating on devset')
    dev_loss = 0.0
    dev_start = time.time()
    refs = {'detection':[], 'resolution':[], 'recovery':[]}
    outputs = {'detection':[], 'resolution':[], 'recovery':[]}
    for step, ori_batch in enumerate(dev_batches):
        # data preparing
        batch = {k: v.to(device) if type(v) == torch.Tensor else v \
                for k, v in ori_batch.items()}
        input_ids, input_mask, input_wordmask, input_char2word, input_char2word_mask = \
                batch['input_ids'], batch['input_mask'], batch['input_wordmask'], \
                batch['input_char2word'], batch['input_char2word_mask']
        input_zp, input_zp_cid, input_zp_span, batch_type = \
                batch['input_zp'], batch['input_zp_cid'], batch['input_zp_span'], batch['type']
        # model execution
        loss, detection_out, recovery_out = model(input_ids, input_mask, input_wordmask, input_char2word, input_char2word_mask,
                input_zp, input_zp_span, input_zp_cid, batch_type)
        dev_loss += loss.item()
        # copy results
        sequence_lengths = input_wordmask.sum(dim=-1).long().cpu().tolist() # [batch]
        copy_batch(input_zp.cpu().tolist(), refs['detection'], sequence_lengths)
        copy_batch(input_zp_cid.cpu().tolist(), refs['recovery'], sequence_lengths)
        copy_batch(detection_out.cpu().tolist(), outputs['detection'], sequence_lengths)
        copy_batch(recovery_out.cpu().tolist(), outputs['recovery'], sequence_lengths)
    # final eval
    print('Dev loss: %.2f, time: %.3f sec' % (dev_loss, time.time()-dev_start))
    det_pr, det_rc, det_f1 = calc_seq_f1(outputs['detection'], refs['detection'])
    print('Detection F1: %.2f, Precision: %.2f, Recall: %.2f' % (100*det_f1, 100*det_pr, 100*det_rc))
    rec_pr, rec_rc, rec_f1 = calc_seq_f1(outputs['recovery'], refs['recovery'])
    print('Recovery F1: %.2f, Precision: %.2f, Recall: %.2f' % (100*rec_f1, 100*rec_pr, 100*rec_rc))
    model.train()
    return det_f1, rec_f1


def main():
    log_dir = FLAGS.log_dir
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    path_prefix = log_dir + "/ZP.{}".format(FLAGS.suffix)
    log_file_path = path_prefix + ".log"
    print('Log file path: {}'.format(log_file_path))
    log_file = open(log_file_path, 'wt')
    log_file.write("{}\n".format(FLAGS))
    log_file.flush()
    config_utils.save_config(FLAGS, path_prefix + ".config.json")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_gpu = torch.cuda.device_count()
    print('device: {}, n_gpu: {}, grad_accum_steps: {}'.format(device, n_gpu, FLAGS.grad_accum_steps))

    tokenizer = BertTokenizer.from_pretrained(FLAGS.bert_model)
    pro_mapping = json.load(open(FLAGS.pro_mapping, 'r'))
    print('Number of predefined pronouns: {}, they are: {}'.format(len(pro_mapping), pro_mapping.values()))

    # load data
    train_features = zp_datastream.load_and_extract_features(FLAGS.train_path, tokenizer,
            data_type="recovery", char2word_strategy="last")
    dev_features = zp_datastream.load_and_extract_features(FLAGS.dev_path, tokenizer,
            data_type="recovery", char2word_strategy="last")
    test_features = zp_datastream.load_and_extract_features(FLAGS.test_path, tokenizer,
            data_type="recovery", char2word_strategy="last")

    # make batches
    print('Making batches')
    train_batches = zp_datastream.make_recovery_batch(train_features, FLAGS.batch_size,
            is_sort=FLAGS.is_sort, is_shuffle=FLAGS.is_shuffle)
    dev_batches = zp_datastream.make_recovery_batch(dev_features, FLAGS.batch_size,
            is_sort=FLAGS.is_sort, is_shuffle=FLAGS.is_shuffle)
    test_batches = zp_datastream.make_recovery_batch(test_features, FLAGS.batch_size,
            is_sort=FLAGS.is_sort, is_shuffle=FLAGS.is_shuffle)

    print("Num training examples = {}".format(len(train_features)))
    print("Num training batches = {}".format(len(train_batches)))
    print("Num dev examples = {}".format(len(dev_features)))
    print("Num dev batches = {}".format(len(dev_batches)))
    print("Num test examples = {}".format(len(test_features)))
    print("Num test batches = {}".format(len(test_batches)))

    # create model
    print('Compiling model')
    model = zp_model.BertZeroProMTL.from_pretrained(FLAGS.bert_model,
            char2word=FLAGS.char2word, pro_num=len(pro_mapping))
    model.to(device)
    if n_gpu > 1:
        model = nn.DataParallel(model)

    print('Starting the training loop, ', end="")
    train_steps = len(train_batches) * FLAGS.num_epochs
    if FLAGS.grad_accum_steps > 1:
        train_steps = train_steps // FLAGS.grad_accum_steps
    print("total steps = {}".format(train_steps))

    named_params = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
    grouped_params = [
            {'params': [p for n, p in named_params if not any(nd in n for nd in no_decay)], 'weight_decay': 0.01},
            {'params': [p for n, p in named_params if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}]
    #grouped_params = [{'params': [p for n, p in named_params], 'weight_decay': 0.0}]
    optimizer = BertAdam(grouped_params,
            lr=FLAGS.learning_rate,
            warmup=FLAGS.warmup_proportion,
            t_total=train_steps)

    best_f1 = 0.0
    global_step = 0
    train_batch_ids = list(range(0, len(train_batches)))
    model.train()
    for _ in range(FLAGS.num_epochs):
        train_loss = 0
        epoch_start = time.time()
        if FLAGS.is_shuffle:
            random.shuffle(train_batch_ids)
        for id in train_batch_ids:
            ori_batch = train_batches[id]
            batch = {k: v.to(device) if type(v) == torch.Tensor else v \
                    for k, v in ori_batch.items()}
            input_ids, input_mask, input_wordmask, input_char2word, input_char2word_mask = \
                    batch['input_ids'], batch['input_mask'], batch['input_wordmask'], \
                    batch['input_char2word'], batch['input_char2word_mask']
            input_zp, input_zp_cid, input_zp_span, batch_type = \
                    batch['input_zp'], batch['input_zp_cid'], batch['input_zp_span'], batch['type']

            loss, _, _ = model(input_ids, input_mask, input_wordmask, input_char2word, input_char2word_mask,
                    input_zp, input_zp_span, input_zp_cid, batch_type)

            if n_gpu > 1:
                loss = loss.mean()
            if FLAGS.grad_accum_steps > 1:
                loss = loss / FLAGS.grad_accum_steps
            train_loss += loss.item()

            loss.backward() # just calculate gradient
            global_step += 1

            train_loss += loss.item()
            if global_step % FLAGS.grad_accum_steps == 0:
                optimizer.step()
                optimizer.zero_grad()

            if global_step % 100 == 0:
                print('{} '.format(global_step), end="")
                sys.stdout.flush()

        print('\nTraining loss: %.2f, time: %.3f sec' % (train_loss, time.time()-epoch_start))
        detection_f1, recovery_f1 = dev_eval(model, device, dev_batches)
        if recovery_f1 > best_f1:
            print('Saving weights, F1 {} (prev_best) < {} (cur)'.format(best_f1, recovery_f1))
            best_f1 = recovery_f1
            save_model(model, path_prefix)
        print('-------------')
        dev_eval(model, device, test_batches)
        print('=============')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_model(model, path_prefix):
    model_to_save = model.module if hasattr(model, 'module') else model

    model_path = path_prefix + ".bert_model.bin"
    config_path = path_prefix + ".bert_config.json"

    torch.save(model_to_save.state_dict(), model_path)
    model_to_save.config.to_json_file(config_path)


def enrich_options():
    pass


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config_path', type=str, help='Configuration file.')

    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   # see issue #152
    os.environ["CUDA_VISIBLE_DEVICES"]="0"

    print("CUDA_VISIBLE_DEVICES " + os.environ['CUDA_VISIBLE_DEVICES'])
    FLAGS, unparsed = parser.parse_known_args()

    if FLAGS.config_path is not None:
        print('Loading hyperparameters from ' + FLAGS.config_path)
        FLAGS = config_utils.load_config(FLAGS.config_path)

    enrich_options()
    assert type(FLAGS.grad_accum_steps) == int and FLAGS.grad_accum_steps >= 1

    main()

