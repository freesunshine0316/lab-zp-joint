
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

FLAGS = None


def calc_f1(n_out, n_ref, n_both):
    #print('n_out {}, n_ref {}, n_both {}'.format(n_out, n_ref, n_both))
    pr = n_both/n_out if n_out > 0.0 else 0.0
    rc = n_both/n_ref if n_ref > 0.0 else 0.0
    f1 = 2.0*pr*rc/(pr+rc) if pr > 0.0 and rc > 0.0 else 0.0
    return pr, rc, f1


# [0,0] or 0 mean 'not applicable'
def add_counts(out, ref, counts):
    assert type(out) in (int, list)
    assert type(ref) in (int, list)
    if type(out) == int:
        out = [out,]
    if type(ref) == int:
        ref = [ref,]
    if sum(out) != 0:
        counts[1] += 1.0
    if sum(ref) != 0:
        counts[2] += 1.0
        if out == ref:
            counts[0] += 1.0


def add_counts_bow(out, ref, counts):
    if sum(out) != 0:
        counts[1] += out[1] - out[0] + 1
    if sum(ref) != 0:
        counts[2] += ref[1] - ref[0] + 1
        ins_st, ins_ed = max(out[0],ref[0]), min(out[1],ref[1])
        count[0] += max(ins_ed - ins_st + 1, 0)


def dev_eval(model, model_type, dev_batches, device, log_file):
    model.eval()
    dev_eval_results = []
    for devset in dev_batches:
        data_type = devset['data_type']
        batches = devset['batches']
        assert data_type in ('recovery', 'resolution')
        print('Evaluating on dataset with data_type: {}'.format(data_type))
        N = 0
        total_loss = 0.0
        counts = {'detection':[0.0 for _ in range(3)],
                'recovery':[0.0 for _ in range(3)],
                'resolution':[0.0 for _ in range(3)],
                'resolution_bow':[0.0 for _ in range(3)]}
        start = time.time()
        for step, ori_batch in enumerate(batches):
            # execution
            batch = {k: v.to(device) if type(v) == torch.Tensor else v \
                    for k, v in ori_batch.items()}
            loss, detection_out, tmp_out = forward_step(model,
                    model_type, batch)
            input_zp, input_zp_cid, input_zp_span, input_ci2wi = \
                    batch['input_zp'], batch['input_zp_cid'], batch['input_zp_span'], batch['input_ci2wi']
            input_zp, detection_out = input_zp.cpu().tolist(), detection_out.cpu().tolist()
            if data_type == 'recovery':
                input_zp_cid = input_zp_cid.cpu().tolist()
                recovery_out = tmp_out.cpu().tolist()
            else:
                input_zp_span = input_zp_span.cpu().tolist()
                resolution_out = tmp_out.cpu().tolist()
            total_loss += loss.item()
            # generate results and counts for F1
            if model_type == 'bert_char': # if char-level model
                mask = batch['input_decision_mask']
                lens = batch['input_mask'].sum(dim=-1).long()
            else:
                mask = batch['input_wordmask']
                lens = batch['input_wordmask'].sum(dim=-1).long()
            B = list(lens.size())[0]
            for i in range(B):
                for j in range(1, lens[i]-1): # [CLS] A B C ... [SEP]
                    # for bert-char model, only consider word-boundary positions
                    # for word models, every position within 'input_wordmask' need to be considered
                    if mask[i,j] == 0.0:
                        continue
                    add_counts(out=detection_out[i][j], ref=input_zp[i][j],
                            counts=counts['detection'])
                    if data_type == 'recovery':
                        add_counts(out=recovery_out[i][j], ref=input_zp_cid[i][j],
                                counts=counts['recovery'])
                    else:
                        add_counts(out=resolution_out[i][j], ref=input_zp_span[i][j],
                                counts=counts['resolution'])
                        out = resolution_out[i][j]
                        out = input_ci2wi[i][out[0]], input_ci2wi[i][out[1]]
                        ref = input_sp_span[i][j]
                        ref = input_ci2wi[i][ref[0]], input_ci2wi[i][ref[1]]
                        add_counts_bow(out=out, ref=ref,
                                counts=counts['resolution_bow'])
                N += B
        # output and calculate performance
        duration = time.time()-start
        print('Loss: %.2f, time: %.3f sec' % (total_loss, duration))
        log_file.write('Loss: %.2f, time: %.3f sec\n' % (total_loss, duration))
        det_pr, det_rc, det_f1 = calc_f1(n_out=counts['detection'][1],
                n_ref=counts['detection'][2], n_both=counts['detection'][0])
        print('Detection F1: %.2f, Precision: %.2f, Recall: %.2f' % (100*det_f1, 100*det_pr, 100*det_rc))
        log_file.write('Detection F1: %.2f, Precision: %.2f, Recall: %.2f\n' % (100*det_f1, 100*det_pr, 100*det_rc))
        cur_result = {'data_type':data_type, 'loss':total_loss, 'detection_f1':det_f1}
        if data_type == 'recovery':
            rec_pr, rec_rc, rec_f1 = calc_f1(n_out=counts['recovery'][1],
                    n_ref=counts['recovery'][2], n_both=counts['recovery'][0])
            print('Recovery F1: %.2f, Precision: %.2f, Recall: %.2f' % (100*rec_f1, 100*rec_pr, 100*rec_rc))
            log_file.write('Recovery F1: %.2f, Precision: %.2f, Recall: %.2f\n' % (100*rec_f1, 100*rec_pr, 100*rec_rc))
            cur_result['key_f1'] = rec_f1
        else:
            res_pr, res_rc, res_f1 = calc_f1(n_out=counts['resolution'][1],
                    n_ref=counts['resolution'][2], n_both=counts['resolution'][0])
            print('Resolution F1: %.2f, Precision: %.2f, Recall: %.2f' % (100*res_f1, 100*res_pr, 100*res_rc))
            log_file.write('Resolution F1: %.2f, Precision: %.2f, Recall: %.2f\n' % (100*res_f1, 100*res_pr, 100*res_rc))
            cur_result['key_f1'] = res_f1
            bow_pr, bow_rc, bow_f1 = calc_f1(n_out=counts['resolution_bow'][1],
                    n_ref=counts['resolution_bow'][2], n_both=counts['resolution_bow'][0])
            print('Resolution BoW F1: %.2f, Precision: %.2f, Recall: %.2f' % (100*bow_f1, 100*bow_pr, 100*bow_rc))
            log_file.write('Resolution BoW F1: %.2f, Precision: %.2f, Recall: %.2f\n' % (100*bow_f1, 100*bow_pr, 100*bow_rc))
            cur_result['resolution_bow_f1'] = bow_f1
        log_file.flush()
        dev_eval_results.append(cur_result)
    model.train()
    return dev_eval_results


def forward_step(model, model_type, batch):
    if model_type == 'bert_word':
        input_ids, input_mask, input_wordmask, input_char2word, input_char2word_mask = \
                batch['input_ids'], batch['input_mask'], batch['input_wordmask'], \
                batch['input_char2word'], batch['input_char2word_mask']
        input_zp, input_zp_cid, input_zp_span, batch_type = \
                batch['input_zp'], batch['input_zp_cid'], batch['input_zp_span'], batch['type']

        loss, out1, out2 = model(input_ids, input_mask, input_wordmask, input_char2word, input_char2word_mask,
                input_zp, input_zp_span, input_zp_cid, batch_type)
    elif model_type == 'bert_char':
        input_ids, input_mask, input_decision_mask = \
                batch['input_ids'], batch['input_mask'], batch['input_decision_mask']
        input_zp, input_zp_cid, input_zp_span, batch_type = \
                batch['input_zp'], batch['input_zp_cid'], batch['input_zp_span'], batch['type']

        loss, out1, out2 = model(input_ids, input_mask, input_decision_mask,
                input_zp, input_zp_span, input_zp_cid, batch_type)
    else:
        assert False, "model_type '{}' not supported".format(model_type)
    return loss, out1, out2


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

    tokenizer = None
    if 'bert' in FLAGS.pretrained_path:
        tokenizer = BertTokenizer.from_pretrained(FLAGS.pretrained_path)

    pro_mapping = json.load(open(FLAGS.pro_mapping, 'r'))
    print('Number of predefined pronouns: {}, they are: {}'.format(len(pro_mapping), pro_mapping.values()))

    # load data and make_batches
    print('Loading data and making batches')
    train_instance_size = 0
    train_batches = []
    train_type_ranges = []
    for path, data_type in zip(FLAGS.train_path, FLAGS.train_type):
        features = zp_datastream.load_and_extract_features(path, tokenizer,
                char2word=FLAGS.char2word, data_type=data_type)
        batches = zp_datastream.make_batch(data_type, features, FLAGS.batch_size,
                is_sort=FLAGS.is_sort, is_shuffle=FLAGS.is_shuffle)
        train_instance_size += len(features)
        train_batches.extend(batches)
        train_type_ranges.append(len(train_batches))

    devsets = []
    for path, data_type in zip(FLAGS.dev_path, FLAGS.dev_type):
        features = zp_datastream.load_and_extract_features(path, tokenizer,
                char2word=FLAGS.char2word, data_type=data_type)
        batches = zp_datastream.make_batch(data_type, features, FLAGS.batch_size,
                is_sort=FLAGS.is_sort, is_shuffle=FLAGS.is_shuffle)
        devsets.append({'data_type':data_type, 'batches':batches})

    # TODO: multi-testsets, like devset setting
    test_features = zp_datastream.load_and_extract_features(FLAGS.test_path, tokenizer,
            char2word=FLAGS.char2word, data_type=FLAGS.test_type)
    test_batches = zp_datastream.make_batch(FLAGS.test_type, test_features, FLAGS.batch_size,
            is_sort=FLAGS.is_sort, is_shuffle=FLAGS.is_shuffle)
    testset = {'data_type':FLAGS.test_type, 'batches':test_batches}

    print("Num training examples = {}".format(train_instance_size))
    print("Num training batches = {}".format(len(train_batches)))

    # create model
    print('Compiling model')
    model = zp_model.BertZP.from_pretrained(FLAGS.pretrained_path,
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

            loss, _, _ = forward_step(model, FLAGS.model_type, batch)

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

        duration = time.time()-epoch_start
        print('\nTraining loss: %.2f, time: %.3f sec' % (train_loss, duration))
        log_file.write('\nTraining loss: %.2f, time: %.3f sec\n' % (train_loss, duration))
        cur_f1 = None
        for dev_result in dev_eval(model, FLAGS.model_type, devsets, device, log_file):
            if FLAGS.dev_type_key == dev_result['data_type']:
                cur_f1 = dev_result['key_f1']
        if cur_f1 > best_f1:
            print('Saving weights, F1 {} (prev_best) < {} (cur)'.format(best_f1, cur_f1))
            log_file.write('Saving weights, F1 {} (prev_best) < {} (cur)\n'.format(best_f1, cur_f1))
            best_f1 = cur_f1
            save_model(model, path_prefix)
        print('-------------')
        log_file.write('-------------\n')
        dev_eval(model, FLAGS.model_type, [testset], device, log_file)
        print('=============')
        log_file.write('=============\n')
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def save_model(model, path_prefix):
    model_to_save = model.module if hasattr(model, 'module') else model

    model_path = path_prefix + ".bert_model.bin"
    model_config_path = path_prefix + ".bert_config.json"

    torch.save(model_to_save.state_dict(), model_path)
    model_to_save.config.to_json_file(model_config_path)


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

    assert type(FLAGS.grad_accum_steps) == int and FLAGS.grad_accum_steps >= 1

    if FLAGS.model_type == 'bert_word':
        import zp_datastream
        import zp_model
    elif FLAGS.model_type == 'bert_char':
        import zp_datastream_char as zp_datastream
        import zp_model_char as zp_model
    else:
        assert False, "model_type '{}' not supported".format(FLAGS.model_type)

    main()

