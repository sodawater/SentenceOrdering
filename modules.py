from __future__ import print_function
import tensorflow as tf
import numpy as np
from tensorflow.python.ops import array_ops

def index_matrix_to_pairs(index_matrix):
  # [[3,1,2], [2,3,1]] -> [[[0, 3], [1, 1], [2, 2]],
  #                        [[0, 2], [1, 3], [2, 1]]]
  replicated_first_indices = tf.range(tf.shape(index_matrix)[0])
  rank = len(index_matrix.get_shape())
  if rank == 2:
    replicated_first_indices = tf.tile(
        tf.expand_dims(replicated_first_indices, dim=1),
        [1, tf.shape(index_matrix)[1]])
  return tf.stack([replicated_first_indices, index_matrix], axis=rank)

def reverse(input_, seq_lengths, seq_dim, batch_dim):
    if seq_lengths is not None:
        return array_ops.reverse_sequence(
            input=input_, seq_lengths=seq_lengths,
            seq_dim=seq_dim, batch_dim=batch_dim)
    else:
        return array_ops.reverse(input_, axis=[seq_dim])

def normalize(inputs,
              epsilon=1e-8,
              scope="ln",
              reuse=None):

    with tf.variable_scope(scope, reuse=reuse):
        inputs_shape = inputs.get_shape()
        params_shape = inputs_shape[-1:]

        mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
        beta = tf.Variable(tf.zeros(params_shape))
        gamma = tf.Variable(tf.ones(params_shape))
        normalized = (inputs - mean) / ((variance + epsilon) ** (.5))
        outputs = gamma * normalized + beta

    return outputs


def multihead_attention(queries,
                        keys,
                        sequence_length,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        causality=False,
                        pointer=False,
                        residual=True,
                        using_mask=False,
                        mymasks=None,
                        scope="multihead_attention",
                        reuse=None):
    with tf.variable_scope(scope, reuse=reuse):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)

        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)


        outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))


        outputs = outputs / (K_.get_shape().as_list()[-1] ** 0.5)


        key_masks = tf.sequence_mask(sequence_length, tf.shape(keys)[1], dtype=tf.float32)
        key_masks = tf.tile(key_masks, [num_heads, 1])
        key_masks = tf.tile(tf.expand_dims(key_masks, 1), [1, tf.shape(queries)[1], 1])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
        if pointer == False:
            if causality:
                diag_vals = tf.ones_like(outputs[0, :, :])
                tril = tf.linalg.LinearOperatorLowerTriangular(diag_vals).to_dense()
                masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(outputs)[0], 1, 1])
                paddings = tf.ones_like(masks) * (-2 ** 32 + 1)

                if using_mask:
                    mymask = tf.tile(tf.expand_dims(mymasks, 0), [tf.shape(outputs)[0], 1, 1])
                    outputs = tf.where(tf.equal(mymask, 0), paddings, outputs)
                else:
                    outputs = tf.where(tf.equal(masks, 0), paddings, outputs)

            outputs = tf.nn.softmax(outputs)
            query_masks = tf.sequence_mask(sequence_length, tf.shape(queries)[1], dtype=tf.float32)
            query_masks = tf.tile(query_masks, [num_heads, 1])
            query_masks = tf.tile(tf.expand_dims(query_masks, -1), [1, 1, tf.shape(keys)[1]])
            outputs *= query_masks

            outputs = tf.matmul(outputs, V_)


            outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)
            outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))

            if residual:
                _, outputs = fusion_gate(outputs, queries)
                outputs = normalize(outputs)

            return outputs
        else:
            return outputs


def w_encoder_attention(queries,
                        keys,
                        sequence_length,
                        num_units=None,
                        num_heads=8,
                        dropout_rate=0,
                        is_training=True,
                        scope="w_encoder_attention",
                        reuse=None):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        Q = tf.layers.dense(queries, num_units, activation=tf.nn.relu)  # (N, T_q, C)
        K = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        V = tf.layers.dense(keys, num_units, activation=tf.nn.relu)  # (N, T_k, C)
        x = K * Q
        x = tf.reshape(x, [tf.shape(x)[0],tf.shape(x)[1],num_heads, int(num_units/num_heads)])
        outputs = tf.transpose(tf.reduce_sum(x, 3),[0,2,1])
        outputs = outputs / (K.get_shape().as_list()[-1] ** 0.5)
        key_masks = tf.sequence_mask(sequence_length, tf.shape(keys)[1], dtype=tf.float32)
        key_masks = tf.reshape(tf.tile(key_masks,[1, num_heads]),[tf.shape(key_masks)[0],num_heads,tf.shape(key_masks)[1]])

        paddings = tf.ones_like(outputs) * (-2 ** 32 + 1)
        outputs = tf.where(tf.equal(key_masks, 0), paddings, outputs)
        outputs = tf.nn.softmax(outputs, 2)
        V_ = tf.reshape(V, [tf.shape(V)[0],tf.shape(V)[1], num_heads, int(num_units/num_heads)])
        V_ = tf.transpose(V_, [0,2,1,3])

        weight = outputs
        outputs = tf.reshape(tf.reduce_sum(V_ * tf.expand_dims(outputs, -1),2),[-1,num_units])
        outputs = tf.layers.dropout(outputs, rate=dropout_rate, training=tf.convert_to_tensor(is_training))
    return outputs, weight



def sigmoid_gate(keys,
                 scope="sigmoid_gate"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        gate = tf.layers.dense(keys, 100, activation=tf.nn.relu)
        gate = tf.layers.dense(gate, 2)
        keys = keys * tf.nn.sigmoid(gate)
        return gate, keys

def fusion_gate(key1,
                key2,
                scope="fusion_gate"):
    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        gate = tf.layers.dense(tf.concat([key1, key2], 2), 1, activation=tf.nn.sigmoid)
        keys = key1 * gate + key2 * (1 - gate)
        return gate, keys

def feedforward(inputs,
                num_units=[2048, 512],
                scope="feedforward",
                reuse=None):

    with tf.variable_scope(scope, reuse=tf.AUTO_REUSE):
        # Inner layer
        params = {"inputs": inputs, "filters": num_units[0], "kernel_size": 1,
                  "activation": tf.nn.relu, "use_bias": True}
        outputs = tf.layers.conv1d(**params)

        # Readout layer
        params = {"inputs": outputs, "filters": num_units[1], "kernel_size": 1,
                  "activation": None, "use_bias": True}
        outputs = tf.layers.conv1d(**params)


        outputs = normalize(outputs)

    return outputs