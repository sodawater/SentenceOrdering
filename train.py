from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import random
import sys
import time
import logging

import numpy as np
#from six.moves import range  # pylint: disable=redefined-builtin
import tensorflow as tf
import os
import shutil
import hashlib
from sys import platform
import data_utils
import argparse
import copy
import collections
from gensim.models import KeyedVectors
from orderingNet import  OrderingNet
FLAGS = None

def add_arguments(parser):
    parser.register("type", "bool", lambda v: v.lower() == "true")
    parser.add_argument("--data_dir", type=str, default="roc/", help="Data directory")
    parser.add_argument("--model_dir", type=str, default="model/", help="Model directory")
    parser.add_argument("--out_dir", type=str, default="output/", help="Out directory")
    parser.add_argument("--gpu_device", type=str, default="2", help="which gpu to use")
    parser.add_argument("--from_train_data", type=str, default="train.src",
                        help="Training data_src path")
    parser.add_argument("--to_train_data", type=str, default="train.dst",
                        help="Training data_dst path")
    parser.add_argument("--train_data", type=str, default="train.ids",
                        help="Training data path")

    parser.add_argument("--from_valid_data", type=str, default="test.src",
                        help="Valid data_src path")
    parser.add_argument("--to_valid_data", type=str, default="test.dst",
                        help="Valid data_dst path")
    parser.add_argument("--valid_data", type=str, default="valid.ids",
                        help="Valid data path")

    parser.add_argument("--from_test_data", type=str, default="test.src",
                        help="Test data_src path")
    parser.add_argument("--to_test_data", type=str, default="test.dst",
                        help="Test data_dst path")
    parser.add_argument("--test_data", type=str, default="test.ids",
                        help="Test data path")

    parser.add_argument("--from_vocab", type=str, default="vocab_20000",
                        help="from vocab path")
    parser.add_argument("--to_vocab", type=str, default="vocab_20000",
                        help="to vocab path")
    parser.add_argument("--output_dir", type=str, default="tfm/")
    parser.add_argument("--ckpt_dir", type=str, default="tfm/",
                        help="model checkpoint directory")
    parser.add_argument("--train_dir", type=str, default="tfm/", help="Training directory")
    parser.add_argument("--from_vocab_size", type=int, default=20000, help="NormalWiki vocabulary size")
    parser.add_argument("--to_vocab_size", type=int, default=20000, help="SimpleWiki vocabulary size")
    parser.add_argument("--num_layers", type=int, default=2, help="Number of layers in the model")
    parser.add_argument("--num_units", type=int, default=256, help="Size of each model layer")
    parser.add_argument("--emb_dim", type=int, default=300, help="Dimension of word embedding")
    parser.add_argument("--num_blocks", type=int, default=3, help="Size of each model blocks")
    parser.add_argument("--num_heads", type=int, default=4, help="Size of each model heads")
    parser.add_argument("--batch_size", type=int, default=64, help="Batch size to use during training")
    parser.add_argument("--max_gradient_norm", type=float, default=3.0, help="Clip gradients to this norm")
    parser.add_argument("--learning_rate_decay_factor", type=float, default=0.5, help="Learning rate decays by this much")
    parser.add_argument("--learning_rate", type=float, default=0.0001, help="Learning rate")

    parser.add_argument("--input_keep_prob", type=float, default=0.85, help="Dropout input keep prob")
    parser.add_argument("--output_keep_prob", type=float, default=1.0, help="Dropout output keep prob")




def safe_exp(value):
  """Exponentiation with catching of overflow error."""
  try:
    ans = math.exp(value)
  except OverflowError:
    ans = float("inf")
  return ans

def get_config_proto(log_device_placement=False, allow_soft_placement=True):
  # GPU options:
  # https://www.tensorflow.org/versions/r0.10/how_tos/using_gpu/index.html
  config_proto = tf.ConfigProto(
      log_device_placement=log_device_placement,
      allow_soft_placement=allow_soft_placement)
  config_proto.gpu_options.allow_growth = True
  return config_proto


class TrainModel(
    collections.namedtuple("TrainModel",
                           ("graph", "model"))):
  pass

class EvalModel(
    collections.namedtuple("EvalModel",
                           ("graph", "model"))):
  pass

class InferModel(
    collections.namedtuple("InferModel",
                           ("graph", "model"))):
  pass

def create_model(hparams, model, length=22):
    train_graph = tf.Graph()
    with train_graph.as_default():
        train_model = model(hparams, tf.contrib.learn.ModeKeys.TRAIN)

    eval_graph = tf.Graph()
    with eval_graph.as_default():
        eval_model = model(hparams, tf.contrib.learn.ModeKeys.EVAL)

    infer_graph = tf.Graph()
    with infer_graph.as_default():
        infer_model = model(hparams, tf.contrib.learn.ModeKeys.INFER)

    return TrainModel(graph=train_graph, model=train_model), EvalModel(graph=eval_graph, model=eval_model), InferModel(
        graph=infer_graph, model=infer_model)


def read_data(src_path, vocab_path):
    data_set = []
    max_length1, max_length2 = 0, 0
    from_vocab, rev_from_vocab = data_utils.initialize_vocabulary(vocab_path)
    with tf.gfile.GFile(src_path, mode="r") as src_file:
        src = src_file.readline()
        counter = 0
        while src:
            if counter % 100000 == 0:
                print("  reading data line %d" % counter)
                sys.stdout.flush()
            # if counter > 100000:
            #      break
            sentences = []
            s = []
            for x in src.split(" "):
                id = int(x)
                if id != -1:
                    s.append(id)
                else:
                    if len(s) > max_length1:
                        max_length1 = len(s)
                    if len(s) > 25:
                        s = s[:25]
                    sentences.append(s)
                    s = []
            data_set.append(sentences)
            counter += 1
            src = src_file.readline()
    print(counter)
    print(max_length1)
    return data_set


def getFileLineNums(filename):
    f = open(filename, 'r')
    count = 0
    for line in f:
        count += 1
    return count


def prepend_line(infile, outfile, line):
    with open(infile, 'r') as old:
        with open(outfile, 'w') as new:
            new.write(str(line) + "\n")
            shutil.copyfileobj(old, new)


def prepend_slow(infile, outfile, line):
    with open(infile, 'r') as fin:
        with open(outfile, 'w') as fout:
            fout.write(line + "\n")
            for line in fin:
                fout.write(line)

def train(hparams, train=True, interact=False):

    #embeddings = init_embedding(hparams)
    #hparams.add_hparam(name="embeddings", value=embeddings)
	# pretrained Glove vector
	
    hparams.add_hparam(name="ckpt_path", value=os.path.join(hparams.ckpt_dir, "tfm.ckpt"))


    train_model, eval_model, infer_model = create_model(hparams, OrderingNet)
    config = get_config_proto(
        log_device_placement=False)
    train_sess = tf.Session(config=config, graph=train_model.graph)
    eval_sess = tf.Session(config=config, graph=eval_model.graph)
    infer_sess = tf.Session(config=config, graph=infer_model.graph)

    train_set = read_data("%s/%s" % (hparams.data_dir, hparams.train_data),
                          "%s/%s" % (hparams.data_dir, hparams.from_vocab))
    valid_set = read_data("%s/%s" % (hparams.data_dir, hparams.valid_data),
                          "%s/%s" % (hparams.data_dir, hparams.from_vocab))
    test_set = read_data("%s/%s" % (hparams.data_dir, hparams.test_data),
                          "%s/%s" % (hparams.data_dir, hparams.from_vocab))
    ckpt = tf.train.get_checkpoint_state(hparams.ckpt_dir)
    with train_model.graph.as_default():
        if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
            print("Reading model parameters from %s" % ckpt.model_checkpoint_path)
            train_model.model.saver.restore(train_sess, ckpt.model_checkpoint_path)
            eval_model.model.saver.restore(eval_sess, ckpt.model_checkpoint_path)
            infer_model.model.saver.restore(infer_sess, ckpt.model_checkpoint_path)
            global_step = train_model.model.global_step.eval(session=train_sess)
        else:
            train_sess.run(tf.global_variables_initializer())
            global_step = 0
    vocab_path = "%s/%s" % (hparams.data_dir, hparams.from_vocab)


    step_loss, step_time, total_predict_count, total_loss, total_time, avg_loss, avg_time = 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0


    while global_step <= 100000:
        start_time = time.time()
        step_loss, global_step, predict_count = train_model.model.train_step(train_sess, train_set, True)

        total_loss += step_loss * hparams.batch_size
        total_time += (time.time() - start_time)
        total_predict_count += predict_count
        if global_step % 100 == 0:
            ppl = safe_exp(total_loss / total_predict_count)
            avg_loss = total_loss / 100
            avg_time = total_time / 100
            total_loss, total_predict_count, total_time = 0.0, 0.0, 0.0
            print("global step %d   step-time %.2fs  ppl %.2f  loss %.2f" % (global_step, avg_time, ppl, avg_loss))

        if global_step % 1000 == 0:
            train_model.model.saver.save(train_sess, hparams.ckpt_path, global_step=global_step)
            ckpt = tf.train.get_checkpoint_state(hparams.ckpt_dir)
            if ckpt and tf.train.checkpoint_exists(ckpt.model_checkpoint_path):
                eval_model.model.saver.restore(eval_sess, ckpt.model_checkpoint_path)
                infer_model.model.saver.restore(infer_sess, ckpt.model_checkpoint_path)
                print("load eval model.")
            else:
                raise ValueError("ckpt file not found.")
            for id in range(100):
                step_loss, predict_count = eval_model.model.eval_step(eval_sess, valid_set, no_random=True,
                                                        id=id * hparams.batch_size)
                total_loss += step_loss * hparams.batch_size
                total_predict_count += predict_count
            ppl = safe_exp(total_loss / total_predict_count)

            last_ppl = ppl
            avg_loss = total_loss / 100
            avg_count = total_predict_count / 100.0
            total_loss, total_predict_count, total_time = 0.0, 0.0, 0.0
            print("eval  ppl %.2f  loss %.2f  count %.2f" % (ppl, avg_loss, avg_count))

            if global_step >= 8000:
                to_vocab, rev_to_vocab = data_utils.initialize_vocabulary(vocab_path)
                total = 0
                right = 0
                pm = 0
                rm = 0
                fm = 0
                for id in range(0, int(len(valid_set) / hparams.batch_size)):
                    predict, ans = infer_model.model.infer_step(infer_sess, valid_set, no_random=True,
                                                                id=id * hparams.batch_size)
                    # print(ans)
                    # print(predict)
                    for i in range(hparams.batch_size):
                        ref = ans[i][:ans[i].index(0) + 1]
                        l = len(ref) - 1
                        dic = {}
                        for k in range(0, l + 1):
                            dic[ref[k]] = k
                        score = 0
                        # print(predict)
                        pred = predict[i].tolist()[:]
                        # print(ref)
                        # print(pred)
                        if 0 in pred:
                            pred = pred[:pred.index(0)]
                            l2 = len(pred)
                            for k1 in range(0, l2 - 1):
                                for k2 in range(k1 + 1, l2):
                                    if pred[k1] not in dic or pred[k2] not in dic:
                                        continue
                                    if dic[pred[k1]] < dic[pred[k2]]:
                                        score += 1
                            p_score = float(score) * 2 / l / (l - 1)
                            r_score = float(score) * 2 / l2 / (l2 - 1)
                            if score == 0:
                                f_score = 0
                            else:
                                f_score = 2 * p_score * r_score / (p_score + r_score)
                            flag = 1
                            if l != l2:
                                flag = 0
                            else:
                                for k in range(0, len(ref) - 1):
                                    if pred[k] != ref[k]:
                                        flag = 0
                                        break
                        else:
                            p_score = 0
                            r_score = 0
                            f_score = 0
                            flag = 0
                        pm += p_score
                        rm += r_score
                        fm += f_score
                        right += flag
                        total += 1
                print(float(pm) / total)
                print(float(rm) / total)
                print(float(fm) / total)
                print(float(right) / total)
                print("infer done.")


               
def init_embedding(hparams):
    f = open("roc/vocab_20000", "r", encoding="utf-8")
    vocab = []
    for line in f:
        vocab.append(line.rstrip("\n"))
    

    word_vectors = KeyedVectors.load_word2vec_format("roc_vector.txt")
    emb = []
    num = 0
    for i in range(0, len(vocab)):
        word = vocab[i]
        if word in word_vectors:
            num += 1
            emb.append(word_vectors[word])
        else:
            emb.append((0.1 * np.random.random([hparams.emb_dim]) - 0.05).astype(np.float32))

    print(" init embedding finished")
    emb = np.array(emb)
    print(num)
    print(emb.shape)
    return emb



def create_hparams(flags):
    return tf.contrib.training.HParams(
        # dir path
        data_dir=flags.data_dir,
        train_dir=flags.train_dir,
        ckpt_dir=flags.ckpt_dir,
        output_dir=flags.output_dir,

        # data params
        batch_size=flags.batch_size,
        from_vocab_size=flags.from_vocab_size,
        to_vocab_size=flags.to_vocab_size,
        GO_ID=data_utils.GO_ID,
        EOS_ID=data_utils.EOS_ID,
        PAD_ID=data_utils.PAD_ID,
        emb_dim=flags.emb_dim,
        from_train_data=flags.from_train_data,
        to_train_data=flags.to_train_data,
        train_data=flags.train_data,

        from_valid_data=flags.from_valid_data,
        to_valid_data=flags.to_valid_data,
        valid_data=flags.valid_data,

        from_test_data=flags.from_test_data,
        to_test_data=flags.to_test_data,
        test_data=flags.test_data,

        from_vocab=flags.from_vocab,
        to_vocab=flags.to_vocab,
        share_vocab=True,

        # model params
        input_keep_prob=flags.input_keep_prob,
        output_keep_prob=flags.output_keep_prob,
        init_weight=0.1,
        num_units=flags.num_units,
        num_blocks=flags.num_blocks,
        num_heads=flags.num_heads,
        num_layers=flags.num_layers,
        learning_rate=flags.learning_rate,
        clip_value=flags.max_gradient_norm,
        decay_factor=flags.learning_rate_decay_factor,
        max_seq_length=42,
        epoch_step=0,
    )

def main(_):

    hparams = create_hparams(FLAGS)
    train(hparams)


if __name__ == "__main__":
    my_parser = argparse.ArgumentParser()
    add_arguments(my_parser)
    FLAGS, remaining = my_parser.parse_known_args()
    FLAGS.ckpt_dir = FLAGS.model_dir + FLAGS.ckpt_dir
    FLAGS.train_dir = FLAGS.model_dir + FLAGS.train_dir
    FLAGS.output_dir = FLAGS.out_dir + FLAGS.output_dir
    print(FLAGS)
    os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu_device
    tf.app.run()