import tensorflow as tf
import numpy as np
import random
import math
from tensorflow.python.layers import core as layers_core
from modules import *
import math
class OrderingNet():
    def __init__(self, hparams, mode):
        self.hparams = hparams
        self.vocab_size = hparams.from_vocab_size
        self.num_units = hparams.num_units
        self.emb_dim = hparams.emb_dim
        self.num_layers = hparams.num_layers
        self.learning_rate = tf.Variable(float(hparams.learning_rate), trainable=False)
        self.learning_rate_decay_op = self.learning_rate.assign(self.learning_rate * hparams.decay_factor)
        self.clip_value = hparams.clip_value
        self.max_seq_length = 30
        self.max_sen_length = 7
        self.beam_width = 16
        self.init_weight = hparams.init_weight
        self.flag = True
        self.num_blocks = hparams.num_blocks
        self.num_heads = hparams.num_heads
        self.dropout_rate = 1 - hparams.input_keep_prob
        self.mode = mode
        mymask = np.zeros([self.max_seq_length, self.max_seq_length], dtype=np.float32)
        for i in range(0, self.max_sen_length):
            for j in range(0, self.max_sen_length):
                if math.fabs(i - j) <= 4:
                    mymask[i][j] = 1.
        self.mymask = mymask

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.is_training = True
        else:
            self.is_training = False

        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            self.w_enc_ids = tf.placeholder(tf.int32, [self.max_sen_length,None, None])
            self.w_enc_len = tf.placeholder(tf.int32, [self.max_sen_length, None])
            self.s_enc_len = tf.placeholder(tf.int32, [None])
            self.dec_ids = tf.placeholder(tf.int32, [None, None])
            self.dec_len = tf.placeholder(tf.int32, [None])
            self.target = tf.placeholder(tf.int32, [None, None])
            self.weight = tf.placeholder(tf.float32, [None, None])
            # self.is_training = True
            self.batch_size = tf.size(self.s_enc_len)
        else:
            self.w_enc_ids = tf.placeholder(tf.int32, [self.max_sen_length, None, None])
            self.w_enc_len = tf.placeholder(tf.int32, [self.max_sen_length, None])
            self.s_enc_len = tf.placeholder(tf.int32, [None])
            self.dec_ids = tf.placeholder(tf.int32, [None, None])
            self.dec_len = tf.placeholder(tf.int32, [None])
            # self.is_training = False
            self.batch_size = tf.size(self.s_enc_len)

        if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
            self.input_keep_prob = self.hparams.input_keep_prob
            self.output_keep_prob = self.hparams.output_keep_prob
        else:
            self.input_keep_prob = 1.0
            self.output_keep_prob = 1.0

        with tf.variable_scope("embedding") as scope:
            self.embeddings = tf.Variable(self.init_matrix([self.vocab_size, self.emb_dim]))
            # self.embeddings = tf.Variable(hparams.embeddings)

        self.build_graph()

    def build_graph(self):
        w_encode = self.build_w_encoder()
        self.w_encode = w_encode
        s_encode = self.build_s_encoder(w_encode)
        self.s_encode = s_encode
        if self.mode != tf.contrib.learn.ModeKeys.INFER:
            pred_logits = self.build_decoder(w_encode, s_encode)

            self.logits = pred_logits
            crossent = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=self.target, logits=pred_logits)

            self.bloss = crossent * self.weight

            self.loss = tf.reduce_sum(crossent * self.weight) / tf.to_float(
                 self.batch_size)
            if self.mode == tf.contrib.learn.ModeKeys.TRAIN:
                self.global_step = tf.Variable(0, trainable=False)
                with tf.variable_scope("train_op") as scope:
                    # optimizer = tf.train.GradientDescentOptimizer(self.learning_rate)
                    optimizer = tf.train.AdamOptimizer(self.learning_rate, beta2=0.98, epsilon=1e-9)
                    gradients, v = zip(*optimizer.compute_gradients(self.loss))
                    gradients, _ = tf.clip_by_global_norm(gradients, self.clip_value)
                    self.train_op = optimizer.apply_gradients(zip(gradients, v),
                                                              global_step=self.global_step)
        else:
            pred_logits = self.build_decoder(w_encode, s_encode)
            self.probs = tf.nn.softmax(pred_logits, 2)
            self.sample_id = tf.argmax(pred_logits, 2)

        self.saver = tf.train.Saver(tf.global_variables())
    def build_w_encoder(self):
        with tf.variable_scope("w_encoder") as scope:
            w_encode = []
            w_weight = []
            self.w_query = tf.get_variable("w_Q", [1, self.num_units * 2], dtype=tf.float32)
            if self.num_layers > 1:
                cell_fw = [self._single_cell() for _ in range(self.num_layers)]
                cell_bw = [self._single_cell() for _ in range(self.num_layers)]
                for i in range(self.max_sen_length):
                    with tf.name_scope("w_enc%d" % i) as n_scope:
                        enc_inp = tf.nn.embedding_lookup(self.embeddings, self.w_enc_ids[i])

                        output, state_fw, state_bw = tf.contrib.rnn.stack_bidirectional_dynamic_rnn(cells_fw=cell_fw,
                                                                                                    cells_bw=cell_bw,
                                                                                                    inputs=enc_inp,
                                                                                                    dtype=tf.float32,
                                                                                                    sequence_length=self.w_enc_len[i])

                        fw_c, fw_h = state_fw[self.num_layers - 1]
                        bw_c, bw_h = state_bw[self.num_layers - 1]
                        encode = tf.concat((fw_h, bw_h), axis=1)
                        encode, att = w_encoder_attention(self.w_query,
                                                     output,
                                                     self.w_enc_len[i],
                                                     num_units=self.num_units * 2,
                                                     dropout_rate=self.dropout_rate,
                                                     is_training=self.is_training)
                        w_encode.append(encode)
            else:
                cell_fw = self._single_cell()
                cell_bw = self._single_cell()
                for i in range(self.max_sen_length):
                    with tf.name_scope("w_enc%d" % i) as n_scope:
                        enc_inp = tf.nn.embedding_lookup(self.embeddings, self.w_enc_ids[i])

                        output, state = tf.nn.bidirectional_dynamic_rnn(
                            cell_fw=cell_fw,
                            cell_bw=cell_bw,
                            inputs=enc_inp,
                            dtype=tf.float32,
                            sequence_length=self.w_enc_len[i])
                        fw_c, fw_h = state[0]
                        bw_c, bw_h = state[1]
                        encode = tf.concat((fw_h, bw_h), axis=1)
                        fw_output, bw_output = output
                        output = tf.concat([fw_output, bw_output], 2)
                        encode, att = w_encoder_attention(self.w_query,
                                                     output,
                                                     self.w_enc_len[i],
                                                     num_units=self.num_units * 2,
                                                     dropout_rate=self.dropout_rate,
                                                     is_training=self.is_training,)
                        print(encode)

                        w_encode.append(encode)
                        w_weight.append(att)

        # print(tf.reshape(w_encode,[]))
        # print(np.array(w_encode))
        # w_encode = tf.convert_to_tensor(np.array(w_encode))
        w_encode = tf.stack(w_encode)
        w_encode = tf.transpose(w_encode, perm=[1, 0, 2])
        self.att = w_weight
        # w_encode = tf.stop_gradient(w_encode)
        return w_encode

    def build_s_encoder(self, enc_inp):
        with tf.variable_scope("s_encoder") as scope:
            s_enc = enc_inp
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    s_enc = multihead_attention(queries=s_enc,
                                                keys=s_enc,
                                                sequence_length=self.s_enc_len,
                                                num_units=self.num_units * 2,
                                                num_heads=self.num_heads,
                                                dropout_rate=self.dropout_rate,
                                                is_training=self.is_training,
                                                causality=False,
                                                scope="self_attention")
            self.s_enc = s_enc
        return s_enc

    def build_decoder(self, w_encode, s_output):
        with tf.variable_scope("decoder") as scope:
            idx_pairs = index_matrix_to_pairs(self.dec_ids)
            dec_inp = tf.gather_nd(w_encode, idx_pairs)
            s_dec = dec_inp
            for i in range(self.num_blocks):
                with tf.variable_scope("num_blocks_{}".format(i)):
                    s_dec1 = multihead_attention(queries=s_dec,
                                                keys=s_dec,
                                                sequence_length=self.s_enc_len,
                                                num_units=self.num_units * 2,
                                                num_heads=self.num_heads,
                                                dropout_rate=self.dropout_rate,
                                                is_training=self.is_training,
                                                causality=True,
                                                residual=False,
                                                scope="self_attention")
                    s_dec2 = multihead_attention(queries=s_dec,
                                                keys=s_output,
                                                sequence_length=self.s_enc_len,
                                                num_units=self.num_units * 2,
                                                num_heads=self.num_heads,
                                                dropout_rate=self.dropout_rate,
                                                is_training=self.is_training,
                                                causality=False,
                                                residual=False,
                                                scope="vanilla_attention")
                    gate, s_dec = fusion_gate(s_dec1, s_dec2)
                    s_dec = normalize(s_dec)

            with tf.variable_scope("num_blocks_{}".format(self.num_blocks + 1)):
                s_dec = multihead_attention(queries=s_dec,
                                                keys=s_output,
                                                sequence_length=self.s_enc_len,
                                                num_units=self.num_units * 2,
                                                num_heads=1,
                                                dropout_rate=self.dropout_rate,
                                                is_training=self.is_training,
                                                causality=False,
                                                pointer=True,
                                                scope="self_attention")


        return s_dec




    def get_batch(self, data, no_random=False, id=0):
        hparams = self.hparams
        seq_size = self.max_seq_length
        sen_size = self.max_sen_length
        w_enc_ids = []
        w_enc_lens = []
        s_enc_lens = []
        dec_ids = []
        dec_lens = []
        n_target = []
        n_weight = []
        target = []
        weight = []
        alen = len(data)
        sum = 0
        GO = [hparams.GO_ID] + [hparams.EOS_ID] + [hparams.PAD_ID] * (seq_size - 2)
        PAD = [hparams.PAD_ID] * seq_size
        for j in range(hparams.batch_size):
            if no_random:
                x1 = data[(id + j) % alen]
            else:
                x1 = random.choice(data)
            noise = random.choice(random.choice(data))
            if_noise = False
            if random.random() > 1.0:
                if_noise = True
                x = x1 + [noise]
            else:
                x = x1
            l = len(x)
            p = np.arange(1, l + 1)
            # np.random.shuffle(p)
            xs = [PAD] * sen_size
            ls = [2] * sen_size
            xs[0] = GO
            for i in range(0, l):
                xs[p[i]] = x[i] + [hparams.PAD_ID] * (seq_size - len(x[i]))
                ls[p[i]] = len(x[i])
            w_enc_ids.append(xs)
            w_enc_lens.append(ls)
            s_enc_lens.append(l + 1)
            tmp = p.tolist()
            noise_target = [1] * sen_size
            if if_noise:
                p = []
                num = 0
                for i in tmp:
                    if num != l - 1:
                        p.append(i)
                    num += 1
                noise_target[tmp[l - 1]] = 0
            else:
                p = tmp

            t = p + [0] * (sen_size - len(p))
            d = [0] + p + [0] * (sen_size - len(p) - 1)
            n_target.append(noise_target)
            target.append(t)
            w = [1.0] * (len(p) + 1) + [0.0] * (sen_size - len(p) - 1)
            weight.append(w)
            w = [1.0] * (l + 1) + [0.0] * (sen_size - l - 1)
            n_weight.append(w)
            dec_ids.append(d)
            dec_lens.append(sen_size)
            sum += len(p) + 1

        w_enc_ids = np.transpose(np.array(w_enc_ids), [1, 0, 2])
        w_enc_lens = np.transpose(np.array(w_enc_lens), [1, 0])
        return w_enc_ids, w_enc_lens, s_enc_lens, dec_ids, dec_lens, target, weight, sum





    def train_step(self, sess, data, out=False):
        w_enc_ids, w_enc_lens, s_enc_lens, dec_ids, dec_lens, target, weight, sum = self.get_batch(data)
        feed = {
            self.w_enc_ids: w_enc_ids,
            self.w_enc_len: w_enc_lens,
            self.s_enc_len: s_enc_lens,
            self.dec_ids: dec_ids,
            self.dec_len: dec_lens,
            self.target: target,
            self.weight: weight
        }
        loss, global_step, _, logits, bloss = sess.run(
            [self.loss, self.global_step, self.train_op, self.logits, self.bloss], feed_dict=feed)
        return loss, global_step, sum



    def padding(self, input, PAD_ID):
        pad = np.zeros([self.hparams.batch_size, self.max_sen_length - input.shape[1]], np.int32)
        output = np.concatenate([input, pad], axis=1)
        return output


    def eval_step(self, sess, data, no_random=True, id=0):
        w_enc_ids, w_enc_lens, s_enc_lens, dec_ids, dec_lens, target, weight, sum = self.get_batch(data, no_random, id)
        feed = {
            self.w_enc_ids: w_enc_ids,
            self.w_enc_len: w_enc_lens,
            self.s_enc_len: s_enc_lens,
            self.dec_ids: dec_ids,
            self.dec_len: dec_lens,
            self.target: target,
            self.weight: weight
        }
        loss, bloss = sess.run([self.loss, self.bloss], feed_dict=feed)
        return loss, sum


    def infer_step(self, sess, data, no_random=True, id=0):
        w_enc_ids, w_enc_lens, s_enc_lens, dec_ids, dec_lens, target, weight, sum = self.get_batch(data, no_random, id)
        dec_ids = np.zeros_like(np.array(dec_ids), dtype=np.int32)
        for i in range(0, self.max_sen_length - 1):
            feed = {
                self.w_enc_ids: w_enc_ids,
                self.w_enc_len: w_enc_lens,
                self.s_enc_len: s_enc_lens,
                self.dec_ids: dec_ids,
                self.dec_len: dec_lens,
                # self.target: target,
                # self.weight: weight
            }
            sample_id = sess.run(self.sample_id, feed_dict=feed)
            # print(i)
            for batch in range(self.hparams.batch_size):
                # print(i)
                # print(dec_ids[batch])
                # print(sample_id[batch])
                dec_ids[batch][i + 1] = sample_id[batch][i]
        return sample_id, target

    def infer_step_beam(self, sess, data, no_random=True, id=0):
        w_enc_ids, w_enc_lens, s_enc_lens, dec_ids, dec_lens, target, weight, sum = self.get_batch(data, no_random, id)
        dec_ids = np.array(dec_ids)
        feed = {
            self.w_enc_ids: w_enc_ids,
            self.w_enc_len: w_enc_lens,
            self.s_enc_len: s_enc_lens,
            self.dec_ids: dec_ids,
            self.dec_len: dec_lens,
            # self.target: target,
            # self.weight: weight
        }
        # print(target)
        ans = [0] * self.hparams.batch_size
        probs = sess.run(self.probs, feed_dict=feed)
        beam_inputs = np.array([[[0] * self.max_sen_length] * self.hparams.batch_size] * self.beam_width)
        beam_probs = np.array([[-10000000000.0] * self.hparams.batch_size] * self.beam_width)

        for j in range(self.hparams.batch_size):
            for k in range(s_enc_lens[j]):
                beam_inputs[k][j][1] = k
                beam_probs[k][j] = math.log(probs[j][0][k])
            for k in range(s_enc_lens[j], self.beam_width):
                beam_probs[k][j] = math.log(probs[j][0][0])

        for i in range(1, self.max_sen_length - 1):
            all_inputs = []
            all_probs = []
            # print(i)
            # print(beam_inputs)
            # print(beam_probs)
            for j in range(0, self.beam_width):
                feed = {
                    self.w_enc_ids: w_enc_ids,
                    self.w_enc_len: w_enc_lens,
                    self.s_enc_len: s_enc_lens,
                    self.dec_ids: beam_inputs[j],
                    self.dec_len: dec_lens,
                    # self.target: target,
                    # self.weight: weight
                }

                probs = sess.run(self.probs, feed_dict=feed)
                tmp_inputs = []
                tmp_probs = []
                for k in range(self.beam_width):
                    x = beam_inputs[j].copy()
                    y = beam_probs[j].copy()
                    tmp_inputs.append(x)
                    tmp_probs.append(y)

                for k in range(self.hparams.batch_size):
                    for l in range(0, s_enc_lens[k]):
                        tmp_inputs[l][k][i + 1] = l

                        if probs[k][i][l] <= 0:
                            probs[k][i][l] = 1e-7
                        tmp_probs[l][k] += math.log(probs[k][i][l])
                        for x in range(0, i):
                            if tmp_inputs[l][k][x] == l and l != 0:
                                tmp_probs[l][k] += math.log(1e-7)

                    for l in range(s_enc_lens[k], self.beam_width):
                        tmp_probs[l][k] += math.log(probs[k][i][0])
                        tmp_inputs[l][k][i + 1] = 0
                # print(beam_inputs[j])
                # print(tmp_inputs)
                all_inputs.extend(tmp_inputs)
                all_probs.extend(tmp_probs)
                # print("ff")
                # print(all_inputs)
                # print(all_probs)
            # print(np.array(all_inputs).shape)
            all_inputs = np.transpose(np.array(all_inputs), [1, 0, 2])
            all_probs = np.transpose(np.array(all_probs), [1, 0])

            # print("ff")
            # print(all_inputs)
            # print(all_probs)
            for batch in range(self.hparams.batch_size):
                topk = np.argsort(-all_probs[batch])
                for j in range(self.beam_width):
                    beam_probs[j][batch] = all_probs[batch][topk[j]]
                    beam_inputs[j][batch] = all_inputs[batch][topk[j]]
                if s_enc_lens[batch] == i + 1:
                    ans[batch] = beam_inputs[0][batch].copy()
                    # print()


        sample_id = np.array(beam_inputs[0])
        ans = np.array(ans)

        return ans, target

    def _single_cell(self, x=1):
        single_cell = tf.contrib.rnn.BasicLSTMCell(self.num_units * x)
        single_cell = tf.contrib.rnn.DropoutWrapper(single_cell,
                                                    input_keep_prob=self.input_keep_prob,
                                                    output_keep_prob=self.output_keep_prob)
        return single_cell

    def init_matrix(self, shape):
        return tf.random_normal(shape, stddev=0.1)

    def lr_decay(self, sess):
        return sess.run(self.learning_rate_decay_op)