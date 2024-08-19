# -*- coding: utf-8 -*-
import tensorflow as tf
from tensorflow.python.ops.rnn_cell import *
# from tensorflow.python.ops.rnn_cell_impl import  _Linear
from tensorflow.contrib.rnn.python.ops.core_rnn_cell import _Linear
from tensorflow import keras
from tensorflow.python.ops import math_ops
from tensorflow.python.ops import init_ops
from tensorflow.python.ops import array_ops
from tensorflow.python.ops import variable_scope as vs
from keras import backend as K

from tensorflow.contrib import layers


class QAAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
        num_units: int, The number of units in the GRU cell.
        activation: Nonlinearity to use.  Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
        kernel_initializer: (optional) The initializer to use for the weight and
        projection matrices.
        bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(QAAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        new_h = (1. - att_score) * state + att_score * c
        return new_h, new_h


class VecAttGRUCell(RNNCell):
    """Gated Recurrent Unit cell (cf. http://arxiv.org/abs/1406.1078).
    Args:
        num_units: int, The number of units in the GRU cell.
        activation: Nonlinearity to use.  Default: `tanh`.
        reuse: (optional) Python boolean describing whether to reuse variables
        in an existing scope.  If not `True`, and the existing scope already has
        the given variables, an error is raised.
        kernel_initializer: (optional) The initializer to use for the weight and
        projection matrices.
        bias_initializer: (optional) The initializer to use for the bias.
    """

    def __init__(self,
                 num_units,
                 activation=None,
                 reuse=None,
                 kernel_initializer=None,
                 bias_initializer=None):
        super(VecAttGRUCell, self).__init__(_reuse=reuse)
        self._num_units = num_units
        self._activation = activation or math_ops.tanh
        self._kernel_initializer = kernel_initializer
        self._bias_initializer = bias_initializer
        self._gate_linear = None
        self._candidate_linear = None

    @property
    def state_size(self):
        return self._num_units

    @property
    def output_size(self):
        return self._num_units

    def __call__(self, inputs, state, att_score):
        return self.call(inputs, state, att_score)

    def call(self, inputs, state, att_score=None):
        """Gated recurrent unit (GRU) with nunits cells."""
        if self._gate_linear is None:
            bias_ones = self._bias_initializer
            if self._bias_initializer is None:
                bias_ones = init_ops.constant_initializer(1.0, dtype=inputs.dtype)
            with vs.variable_scope("gates"):  # Reset gate and update gate.
                self._gate_linear = _Linear(
                    [inputs, state],
                    2 * self._num_units,
                    True,
                    bias_initializer=bias_ones,
                    kernel_initializer=self._kernel_initializer)

        value = math_ops.sigmoid(self._gate_linear([inputs, state]))
        r, u = array_ops.split(value=value, num_or_size_splits=2, axis=1)

        r_state = r * state
        if self._candidate_linear is None:
            with vs.variable_scope("candidate"):
                self._candidate_linear = _Linear(
                    [inputs, r_state],
                    self._num_units,
                    True,
                    bias_initializer=self._bias_initializer,
                    kernel_initializer=self._kernel_initializer)
        c = self._activation(self._candidate_linear([inputs, r_state]))
        u = (1.0 - att_score) * u
        new_h = u * state + (1 - u) * c
        return new_h, new_h


def prelu(_x, scope=''):
    """parametric ReLU activation"""
    with tf.variable_scope(name_or_scope=scope, default_name="prelu"):
        _alpha = tf.get_variable("prelu_" + scope, shape=_x.get_shape()[-1],
                                 dtype=_x.dtype, initializer=tf.constant_initializer(0.1))
        return tf.maximum(0.0, _x) + _alpha * tf.minimum(0.0, _x)


def calc_auc(raw_arr):
    """Summary

    Args:
        raw_arr (TYPE): Description

    Returns:
        TYPE: Description
    """

    arr = sorted(raw_arr, key=lambda d: d[0], reverse=True)
    pos, neg = 0., 0.
    for record in arr:
        if record[1] == 1.:
            pos += 1
        else:
            neg += 1

    fp, tp = 0., 0.
    xy_arr = []
    for record in arr:
        if record[1] == 1.:
            tp += 1
        else:
            fp += 1
        xy_arr.append([fp / neg, tp / pos])

    auc = 0.
    prev_x = 0.
    prev_y = 0.
    for x, y in xy_arr:
        if x != prev_x:
            auc += ((x - prev_x) * (y + prev_y) / 2.)
            prev_x = x
            prev_y = y

    return auc


def attention(query, facts, attention_size, mask, stag='null', mode='LIST', softmax_stag=1, time_major=False,
              return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])

    mask = tf.equal(mask, tf.ones_like(mask))
    hidden_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    input_size = query.get_shape().as_list()[-1]

    # Trainable parameters
    w1 = tf.Variable(tf.random_normal([hidden_size, attention_size], stddev=0.1))
    w2 = tf.Variable(tf.random_normal([input_size, attention_size], stddev=0.1))
    b = tf.Variable(tf.random_normal([attention_size], stddev=0.1))
    v = tf.Variable(tf.random_normal([attention_size], stddev=0.1))

    with tf.name_scope('v'):
        # Applying fully connected layer with non-linear activation to each of the B*T timestamps;
        #  the shape of `tmp` is (B,T,D)*(D,A)=(B,T,A), where A=attention_size
        tmp1 = tf.tensordot(facts, w1, axes=1)
        tmp2 = tf.tensordot(query, w2, axes=1)
        tmp2 = tf.reshape(tmp2, [-1, 1, tf.shape(tmp2)[-1]])
        tmp = tf.tanh((tmp1 + tmp2) + b)

    # For each of the timestamps its vector of size A from `tmp` is reduced with `v` vector
    v_dot_tmp = tf.tensordot(tmp, v, axes=1, name='v_dot_tmp')  # (B,T) shape
    key_masks = mask  # [B, 1, T]
    # key_masks = tf.expand_dims(mask, 1) # [B, 1, T]
    paddings = tf.ones_like(v_dot_tmp) * (-2 ** 32 + 1)
    v_dot_tmp = tf.where(key_masks, v_dot_tmp, paddings)  # [B, 1, T]
    alphas = tf.nn.softmax(v_dot_tmp, name='alphas')  # (B,T) shape

    # Output of (Bi-)RNN is reduced with attention vector; the result has (B,D) shape
    # output = tf.reduce_sum(facts * tf.expand_dims(alphas, -1), 1)
    output = facts * tf.expand_dims(alphas, -1)
    output = tf.reshape(output, tf.shape(facts))
    # output = output / (facts.get_shape().as_list()[-1] ** 0.5)
    if not return_alphas:
        return output
    else:
        return output, alphas


def din_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                  return_alphas=False, return_att=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print("querry_size mismatch")
        query = tf.concat(values=[
            query,
            query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    
    if return_att:
        return output, scores

    return output


def din_attention_new1(query, facts, context_his_eb, on_size, mask, stag='null', mode='SUM', softmax_stag=1,
                       time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print("querry_size mismatch")
        query = tf.concat(values=[
            query,
            query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    if context_his_eb is None:
        facts = facts
    else:
        fact1 = tf.concat([facts, context_his_eb], axis=-1)
    fact1 = tf.layers.dense(fact1, facts.get_shape().as_list()[-1], activation=None, name=stag + 'dmr1')
    #    fact1= prelu(fact1, scope=stag+'dmr_prelu')
    din_all = tf.concat([queries, fact1, queries - fact1, queries * fact1], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]
    paddings_no_softmax = tf.zeros_like(scores)
    scores_no_softmax = tf.where(key_masks, scores, paddings_no_softmax)

    # Scale
    scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output, scores, scores_no_softmax


def din_attention_new(query, facts, context_his_eb, on_size, mask, stag='null', mode='SUM', softmax_stag=1,
                      time_major=False, return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
        print("querry_size mismatch")
        query = tf.concat(values=[
            query,
            query,
        ], axis=1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    if context_his_eb is None:
        queries = queries
    else:
        queries = tf.concat([queries, context_his_eb], axis=-1)
    queries = tf.layers.dense(queries, facts.get_shape().as_list()[-1], activation=None, name=stag + 'dmr1')
    # queries = prelu(queries, scope=stag+'dmr_prelu')
    # din_all = tf.concat([queries, facts], axis=-1)
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]
    paddings_no_softmax = tf.zeros_like(scores)
    scores_no_softmax = tf.where(key_masks, scores, paddings_no_softmax)

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    return output, scores, scores_no_softmax


def din_fcn_attention(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                      return_alphas=False, forCnn=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, 80, activation=tf.nn.sigmoid, name='f1_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, 40, activation=tf.nn.sigmoid, name='f2_att' + stag)
    d_layer_3_all = tf.layers.dense(d_layer_2_all, 1, activation=None, name='f3_att' + stag)
    d_layer_3_all = tf.reshape(d_layer_3_all, [-1, 1, tf.shape(facts)[1]])
    scores = d_layer_3_all
    # Mask
    # key_masks = tf.sequence_mask(facts_length, tf.shape(facts)[1])   # [B, T]
    key_masks = tf.expand_dims(mask, 1)  # [B, 1, T]
    paddings = tf.ones_like(scores) * (-2 ** 32 + 1)
    if not forCnn:
        scores = tf.where(key_masks, scores, paddings)  # [B, 1, T]

    # Scale
    # scores = scores / (facts.get_shape().as_list()[-1] ** 0.5)

    # Activation
    if softmax_stag:
        scores = tf.nn.softmax(scores)  # [B, 1, T]

    # Weighted sum
    if mode == 'SUM':
        output = tf.matmul(scores, facts)  # [B, 1, H]
        # output = tf.reshape(output, [-1, tf.shape(facts)[-1]])
    else:
        scores = tf.reshape(scores, [-1, tf.shape(facts)[1]])
        output = facts * tf.expand_dims(scores, -1)
        output = tf.reshape(output, tf.shape(facts))
    if return_alphas:
        return output, scores
    return output


def self_attention(facts, ATTENTION_SIZE, mask, stag='null'):
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    def cond(batch, output, i):
        return tf.less(i, tf.shape(batch)[1])

    def body(batch, output, i):
        self_attention_tmp = din_fcn_attention(batch[:, i, :], batch[:, 0:i + 1, :],
                                               ATTENTION_SIZE, mask[:, 0:i + 1], softmax_stag=1, stag=stag,
                                               mode='LIST')
        self_attention_tmp = tf.reduce_sum(self_attention_tmp, 1)
        output = output.write(i, self_attention_tmp)
        return batch, output, i + 1

    output_ta = tf.TensorArray(dtype=tf.float32,
                               size=0,
                               dynamic_size=True,
                               element_shape=(facts[:, 0, :].get_shape()))
    _, output_op, _ = tf.while_loop(cond, body, [facts, output_ta, 0])
    self_attention = output_op.stack()
    self_attention = tf.transpose(self_attention, perm=[1, 0, 2])
    return self_attention


def self_all_attention(facts, ATTENTION_SIZE, mask, stag='null'):
    if len(facts.get_shape().as_list()) == 2:
        facts = tf.expand_dims(facts, 1)

    def cond(batch, output, i):
        return tf.less(i, tf.shape(batch)[1])

    def body(batch, output, i):
        self_attention_tmp = din_fcn_attention(batch[:, i, :], batch,
                                               ATTENTION_SIZE, mask, softmax_stag=1, stag=stag,
                                               mode='LIST')
        self_attention_tmp = tf.reduce_sum(self_attention_tmp, 1)
        output = output.write(i, self_attention_tmp)
        return batch, output, i + 1

    output_ta = tf.TensorArray(dtype=tf.float32,
                               size=0,
                               dynamic_size=True,
                               element_shape=(facts[:, 0, :].get_shape()))
    _, output_op, _ = tf.while_loop(cond, body, [facts, output_ta, 0])
    self_attention = output_op.stack()
    self_attention = tf.transpose(self_attention, perm=[1, 0, 2])
    return self_attention


def din_fcn_shine(query, facts, attention_size, mask, stag='null', mode='SUM', softmax_stag=1, time_major=False,
                  return_alphas=False):
    if isinstance(facts, tuple):
        # In case of Bi-RNN, concatenate the forward and the backward RNN outputs.
        facts = tf.concat(facts, 2)

    if time_major:
        # (T,B,D) => (B,T,D)
        facts = tf.array_ops.transpose(facts, [1, 0, 2])
    # Trainable parameters
    mask = tf.equal(mask, tf.ones_like(mask))
    facts_size = facts.get_shape().as_list()[-1]  # D value - hidden size of the RNN layer
    querry_size = query.get_shape().as_list()[-1]
    query = tf.layers.dense(query, facts_size, activation=None, name='f1_trans_shine' + stag)
    query = prelu(query)
    queries = tf.tile(query, [1, tf.shape(facts)[1]])
    queries = tf.reshape(queries, tf.shape(facts))
    din_all = tf.concat([queries, facts, queries - facts, queries * facts], axis=-1)
    d_layer_1_all = tf.layers.dense(din_all, facts_size, activation=tf.nn.sigmoid, name='f1_shine_att' + stag)
    d_layer_2_all = tf.layers.dense(d_layer_1_all, facts_size, activation=tf.nn.sigmoid, name='f2_shine_att' + stag)
    d_layer_2_all = tf.reshape(d_layer_2_all, tf.shape(facts))
    output = d_layer_2_all
    return output


def Mask(inputs, seq_len, mode='mul'):
    if seq_len == None:
        return inputs
    else:
        mask = tf.cast(tf.sequence_mask(seq_len), tf.float32)
        for _ in range(len(inputs.shape) - 2):
            mask = tf.expand_dims(mask, 2)
        if mode == 'mul':
            return inputs * mask
        if mode == 'add':
            return inputs - (1 - mask) * 1e12


def Dense(inputs, ouput_size, bias=True, seq_len=None):
    input_size = int(inputs.shape[-1])
    W = tf.Variable(tf.random_uniform([input_size, ouput_size], -0.05, 0.05))
    if bias:
        b = tf.Variable(tf.random_uniform([ouput_size], -0.05, 0.05))
    else:
        b = 0
    outputs = tf.matmul(tf.reshape(inputs, (-1, input_size)), W) + b
    outputs = tf.reshape(outputs, tf.concat([tf.shape(inputs)[:-1], [ouput_size]], 0)
                         )
    if seq_len != None:
        outputs = Mask(outputs, seq_len, 'mul')
    return outputs


def self_multi_head_attn(inputs, num_units, num_heads, dropout_rate, name="", is_training=True, is_layer_norm=True, return_att=False):
    """
    Args:
      inputs(query): A 3d tensor with shape of [N, T_q, C_q]
      inputs(keys): A 3d tensor with shape of [N, T_k, C_k]
    """
    Q_K_V = tf.layers.dense(inputs, 3 * num_units)  # tf.nn.relu
    Q, K, V = tf.split(Q_K_V, 3, -1)

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    print('V_.get_shape()', V_.get_shape().as_list())
    # (h*N, T_q, T_k)
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [hN, T, T]
    align = outputs / (36 ** 0.5)
    # align = general_attention(Q_, K_)
    print('align.get_shape()', align.get_shape().as_list())
    diag_val = tf.ones_like(align[0, :, :])  # [T, T]
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [T, T]
    # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense() # [T, T] for tensorflow140
    key_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])
    padding = tf.ones_like(key_masks) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), padding, align)  # [h*N, T, T]
    outputs = tf.nn.softmax(outputs)
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    att = outputs
    outputs = tf.matmul(outputs, V_)
    outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
    # output linear
    outputs = tf.layers.dense(outputs, num_units)

    # drop_out before residual and layernorm
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    # Residual connection
    outputs += inputs  # (N, T_q, C)
    # Normalize
    if is_layer_norm:
        outputs = layer_norm(outputs, name=name)  # (N, T_q, C)

    if return_att:
        return outputs, att
    
    return outputs


def self_multi_head_attn_v2(inputs, num_units, num_heads, dropout_rate, name="", is_training=True, is_layer_norm=True, return_att=False, add=None):
    """
    Args:
      inputs(query): A 3d tensor with shape of [N, T_q, C_q]
      inputs(keys): A 3d tensor with shape of [N, T_k, C_k]
    """
    Q_K_V = tf.layers.dense(inputs, 3 * num_units)  # tf.nn.relu
    Q, K, V = tf.split(Q_K_V, 3, -1)

    Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
    K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
    print('V_.get_shape()', V_.get_shape().as_list())
    # (h*N, T_q, T_k)
    outputs = tf.matmul(Q_, tf.transpose(K_, [0, 2, 1]))  # [hN, T, T]
    align = outputs / (36 ** 0.5)
    # align = general_attention(Q_, K_)
    print('align.get_shape()', align.get_shape().as_list())
    diag_val = tf.ones_like(align[0, :, :])  # [T, T]
    tril = tf.linalg.LinearOperatorLowerTriangular(diag_val).to_dense()  # [T, T]
    # tril = tf.contrib.linalg.LinearOperatorTriL(diag_val).to_dense() # [T, T] for tensorflow140
    key_masks = tf.tile(tf.expand_dims(tril, 0), [tf.shape(align)[0], 1, 1])
    padding = tf.ones_like(key_masks) * (-2 ** 32 + 1)
    outputs = tf.where(tf.equal(key_masks, 0), padding, align)  # [h*N, T, T]

    if add is not None:
        outputs = outputs+add

    att=outputs

    outputs = tf.nn.softmax(outputs)
    outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    outputs = tf.matmul(outputs, V_)
    # Restore shape
    outputs1 = tf.split(outputs, num_heads, axis=0)
    outputs2 = []
    for head_index, outputs3 in enumerate(outputs1):
        outputs3 = tf.layers.dense(outputs3, num_units)
        outputs3 = tf.layers.dropout(outputs3, dropout_rate, training=is_training)
        outputs3 += inputs
        print("outputs3.get_shape()", outputs3.get_shape())
        if is_layer_norm:
            outputs3 = layer_norm(outputs3, name=name + str(head_index))  # (N, T_q, C)
        outputs2.append(outputs3)

    # drop_out before residual and layernorm
    # outputs = tf.layers.dropout(outputs, dropout_rate, training=is_training)
    # Residual connection
    # outputs += inputs  # (N, T_q, C)
    # Normalize
    #  if is_layer_norm:
    #       outputs = layer_norm(outputs)  # (N, T_q, C)
    if return_att:
        return outputs2, att 

    return outputs2


def soft_max_weighted_sum(align, value, key_masks, drop_out, is_training, future_binding=False):
    """
    :param align:           [batch_size, None, time]
    :param value:           [batch_size, time, units]
    :param key_masks:       [batch_size, None, time]
                            2nd dim size with align
    :param drop_out:
    :param is_training:
    :param future_binding:  TODO: only support 2D situation at present
    :return:                weighted sum vector
                            [batch_size, None, units]
    """
    # exp(-large) -> 0
    paddings = tf.fill(tf.shape(align), float('-inf'))
    # [batch_size, None, time]
    align = tf.where(key_masks, align, paddings)

    if future_binding:
        length = tf.reshape(tf.shape(value)[1], [-1])
        # [time, time]
        lower_tri = tf.ones(tf.concat([length, length], axis=0))
        # [time, time]
        lower_tri = tf.contrib.linalg.LinearOperatorTriL(lower_tri).to_dense()
        # [batch_size, time, time]
        masks = tf.tile(tf.expand_dims(lower_tri, 0), [tf.shape(align)[0], 1, 1])
        # [batch_size, time, time]
        align = tf.where(tf.equal(masks, 0), paddings, align)

    # soft_max and dropout
    # [batch_size, None, time]
    align = tf.nn.softmax(align)
    align = tf.layers.dropout(align, drop_out, training=is_training)
    # weighted sum
    # [batch_size, None, units]
    return tf.matmul(align, value)


def general_attention(query, key):
    """
    :param query: [batch_size, None, query_size]
    :param key:   [batch_size, time, key_size]
    :return:      [batch_size, None, time]
        query_size should keep the same dim with key_size
    """
    # [batch_size, None, time]
    align = tf.matmul(query, tf.transpose(key, [0, 2, 1]))
    # scale (optional)
    align = align / (key.get_shape().as_list()[-1] ** 0.5)
    return align


def layer_norm(inputs, name, epsilon=1e-8):
    mean, variance = tf.nn.moments(inputs, [-1], keep_dims=True)
    normalized = (inputs - mean) / (tf.sqrt(variance + epsilon))

    params_shape = inputs.get_shape()[-1:]
    gamma = tf.get_variable(name + 'gamma', params_shape, tf.float32, tf.ones_initializer())
    beta = tf.get_variable(name + 'beta', params_shape, tf.float32, tf.zeros_initializer())

    outputs = gamma * normalized + beta
    return outputs



def Position_Embedding(inputs, position_size):
    batch_size, seq_len = tf.shape(inputs)[0], tf.shape(inputs)[1]
    position_j = 1. / tf.pow(10000., \
                             2 * tf.range(position_size / 2, dtype=tf.float32 \
                                          ) / position_size)
    position_j = tf.expand_dims(position_j, 0)
    position_i = tf.range(tf.cast(seq_len, tf.float32), dtype=tf.float32)
    position_i = tf.expand_dims(position_i, 1)
    position_ij = tf.matmul(position_i, position_j)
    position_ij = tf.concat([tf.cos(position_ij), tf.sin(position_ij)], 1)
    position_embedding = tf.expand_dims(position_ij, 0) \
                         + tf.zeros((batch_size, seq_len, position_size))
    return position_embedding



def multihead_attention(queries,
                        keys,
                        num_units=None,
                        num_output_units=None,
                        activation_fn=None,
                        num_heads=8,
                        scope="multihead_attention",
                        reuse=None,
                        query_masks=None,
                        key_masks=None,
                        atten_mode='base',
                        linear_projection=True,
                        fix_rtp_bug=False,
                        variables_collections=None,
                        outputs_collections=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      num_output_units: A scalar. Output Value size.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
      query_masks: A mask to mask queries with the shape of [N, T_k], if query_masks is None, use queries_length to mask queries
      key_masks: A mask to mask keys with the shape of [N, T_Q],  if key_masks is None, use keys_length to mask keys
      key_projection: A boolean, use projection to keys

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        query_len = queries.get_shape().as_list()[1]  # T_q
        key_len = keys.get_shape().as_list()[1]  # T_k

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        if linear_projection:
            queries_2d = tf.reshape(queries, [-1, queries.get_shape().as_list()[-1]])
            keys_2d = tf.reshape(keys, [-1, keys.get_shape().as_list()[-1]])
            Q = layers.fully_connected(queries_2d,
                                       num_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="Q")  # (N, T_q, C)
            Q = tf.reshape(Q, [-1, queries.get_shape().as_list()[1], Q.get_shape().as_list()[-1]])
            K = layers.fully_connected(keys_2d,
                                       num_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="K")  # (1, T_k, C)
            K = tf.reshape(K, [-1, keys.get_shape().as_list()[1], K.get_shape().as_list()[-1]])
            V = layers.fully_connected(keys_2d,
                                       num_output_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="V")  # (1, T_k, C)
            V = tf.reshape(V, [-1, keys.get_shape().as_list()[1], V.get_shape().as_list()[-1]])
        else:
            Q = queries
            K = keys
            V = keys

        # fix rtp bug
        if fix_rtp_bug:
            K = tf.reshape(K, [tf.shape(Q)[0], tf.shape(K)[1], K.get_shape().as_list()[-1]])
            V = tf.reshape(V, [tf.shape(Q)[0], tf.shape(V)[1], V.get_shape().as_list()[-1]])

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C'/h)

        if atten_mode == 'ln':
            # Layer Norm
            Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
            K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))
        else:
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))

        # key Masking
        key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, key_len]), [num_heads, query_len, 1])  # (h*N, T_q, T_k)
        paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
        outputs = tf.where(key_masks, outputs, paddings)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.tile(tf.reshape(query_masks, [-1, query_len]), [num_heads, 1])  # (h*N, T_q)
        outputs = tf.reshape(outputs, [-1, key_len])  # (h*N*T_q, T_k)
        paddings = tf.zeros_like(outputs, dtype=tf.float32)  # (h*N*T_q, T_k)
        outputs = tf.where(tf.reshape(query_masks, [-1]), outputs,
                           paddings)  # tf.where((h*N*T_q), (h*N*T_q, T_k), (h*N*T_q, T_k)) => (h*N*T_q, T_k)
        outputs = tf.reshape(outputs, [-1, query_len, key_len])  # (h*N, T_q, T_k)

        # Attention vector
        att_vec = outputs

        # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C/h)
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    return outputs, att_vec


def multihead_target_attention(queries,
                               keys,
                               values,
                               num_units=None,
                               num_output_units=None,
                               activation_fn=None,
                               num_heads=8,
                               scope="multihead_attention",
                               reuse=None,
                               key_masks=None,
                               atten_mode='base',
                               linear_projection=True,
                               fix_rtp_bug=False,
                               variables_collections=None,
                               outputs_collections=None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      num_output_units: A scalar. Output Value size.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
      query_masks: A mask to mask queries with the shape of [N, T_k], if query_masks is None, use queries_length to mask queries
      key_masks: A mask to mask keys with the shape of [N, T_Q],  if key_masks is None, use keys_length to mask keys
      key_projection: A boolean, use projection to keys

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        if atten_mode == 'ln':
            activation_fn = None

        query_len = queries.get_shape().as_list()[1]  # T_q
        key_len = keys.get_shape().as_list()[1]  # T_k

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        if linear_projection:
            queries_2d = tf.reshape(queries, [-1, queries.get_shape().as_list()[-1]])
            keys_2d = tf.reshape(keys, [-1, keys.get_shape().as_list()[-1]])
            values_2d = tf.reshape(values, [-1, values.get_shape().as_list()[-1]])
            Q = layers.fully_connected(queries_2d,
                                       num_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="Q")  # (N, T_q, C)
            Q = tf.reshape(Q, [-1, queries.get_shape().as_list()[1], Q.get_shape().as_list()[-1]])
            K = layers.fully_connected(keys_2d,
                                       num_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="K")  # (1, T_k, C)
            K = tf.reshape(K, [-1, tf.shape(keys)[1], tf.shape(K)[-1]])
            V = layers.fully_connected(values_2d,
                                       num_output_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="V")  # (1, T_k, C)
            V = tf.reshape(V, [-1, values.get_shape().as_list()[1], V.get_shape().as_list()[-1]])
        else:
            Q = queries
            K = keys
            V = values

        # fix rtp bug
        if fix_rtp_bug:
            K = tf.reshape(K, [tf.shape(Q)[0], tf.shape(K)[1], K.get_shape().as_list()[-1]])
            V = tf.reshape(V, [tf.shape(Q)[0], tf.shape(V)[1], V.get_shape().as_list()[-1]])

        # Split and concat
        print("multihead target attention Q size: {}".format(str(Q.get_shape().as_list)))
        Q_ = tf.concat(tf.split(Q, num_heads, axis=2), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=2), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=2), axis=0)  # (h*N, T_k, C'/h)

        # Multiplication & Scale
        if atten_mode == 'ln':
            # Layer Norm
            Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
            K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))
        else:
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))

        # key Masking
        key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, key_len]), [num_heads, query_len, 1])  # (h*N, T_q, T_k)
        paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
        outputs = tf.where(key_masks, outputs, paddings)

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Attention vector
        att_vec = outputs

        # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C/h)
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)

    return outputs, att_vec

def multihead_attention2(queries,
                        keys,
                        num_units=None,
                        num_output_units=None,
                        activation_fn=None,
                        num_heads=8,
                        scope="",
                        reuse=None,
                        query_masks=None,
                        key_masks=None,
                        atten_mode='base',
                        linear_projection=True,
                        fix_rtp_bug=False,
                        variables_collections=None,
                        outputs_collections=None,
                        res_connect = False,
                        final_norm = False,
                        add = None):
    '''Applies multihead attention.

    Args:
      queries: A 3d tensor with shape of [N, T_q, C_q].
      keys: A 3d tensor with shape of [N, T_k, C_k].
      num_units: A scalar. Attention size.
      num_output_units: A scalar. Output Value size.
      num_heads: An int. Number of heads.
      scope: Optional scope for `variable_scope`.
      reuse: Boolean, whether to reuse the weights of a previous layer
      by the same name.
      query_masks: A mask to mask queries with the shape of [N, T_k], if query_masks is None, use queries_length to mask queries
      key_masks: A mask to mask keys with the shape of [N, T_Q],  if key_masks is None, use keys_length to mask keys
      key_projection: A boolean, use projection to keys

    Returns
      A 3d tensor with shape of (N, T_q, C)
    '''
    with tf.variable_scope(scope, reuse=reuse):
        # Set the fall back option for num_units
        if num_units is None:
            num_units = queries.get_shape().as_list[-1]

        query_len = queries.get_shape().as_list()[-2]  # T_q
        key_len = keys.get_shape().as_list()[-2]  # T_k

        # Linear projections, C = # dim or column, T_x = # vectors or actions
        if linear_projection:
            queries_2d = tf.reshape(queries, [-1, queries.get_shape().as_list()[-1]])
            keys_2d = tf.reshape(keys, [-1, keys.get_shape().as_list()[-1]])
            Q = layers.fully_connected(queries_2d,
                                       num_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="Q")  # (N, T_q, C)
            Q = tf.reshape(Q, [-1, queries.get_shape().as_list()[-2], Q.get_shape().as_list()[-1]])
            K = layers.fully_connected(keys_2d,
                                       num_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="K")  # (1, T_k, C)
            K = tf.reshape(K, [-1, keys.get_shape().as_list()[-2], K.get_shape().as_list()[-1]])
            V = layers.fully_connected(keys_2d,
                                       num_output_units,
                                       activation_fn=activation_fn,
                                       variables_collections=variables_collections,
                                       outputs_collections=outputs_collections, scope="V")  # (1, T_k, C)
            V = tf.reshape(V, [-1, keys.get_shape().as_list()[-2], V.get_shape().as_list()[-1]])
        else:
            Q = queries
            K = keys
            V = keys

        # Split and concat
        Q_ = tf.concat(tf.split(Q, num_heads, axis=-1), axis=0)  # (h*N, T_q, C/h)
        K_ = tf.concat(tf.split(K, num_heads, axis=-1), axis=0)  # (h*N, T_k, C/h)
        V_ = tf.concat(tf.split(V, num_heads, axis=-1), axis=0)  # (h*N, T_k, C'/h)

        if atten_mode == 'ln':
            # Layer Norm
            Q_ = layers.layer_norm(Q_, begin_norm_axis=-1, begin_params_axis=-1)
            K_ = layers.layer_norm(K_, begin_norm_axis=-1, begin_params_axis=-1)
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))
        else:
            # Multiplication
            outputs = tf.matmul(Q_, K_, transpose_b=True)  # (h*N, T_q, T_k)
            # Scale
            outputs = outputs * (K_.get_shape().as_list()[-1] ** (-0.5))

        # add
        if add is not None:
            outputs = outputs*add

        # key Masking 
        key_masks = tf.cast(key_masks, tf.bool)
        if len(key_masks.get_shape().as_list()) == len(keys.get_shape().as_list())-1:
            #N * T_k
            key_masks = tf.tile(tf.reshape(key_masks, [-1, 1, key_len]), [num_heads, query_len, 1])  # (h*N, T_q, T_k)
        else:
            #N * T_k * T_q
            key_masks = tf.tile(tf.reshape(key_masks, tf.concat([[-1], tf.shape(key_masks)[-2:]], axis=0)), [num_heads, 1, 1])

        paddings = tf.fill(tf.shape(outputs), tf.constant(-2 ** 32 + 1, dtype=tf.float32))
        outputs = tf.where(key_masks, outputs, paddings)
        att_vec = outputs

        # Attention vector

        # Activation
        outputs = tf.nn.softmax(outputs)  # (h*N, T_q, T_k)

        # Query Masking
        query_masks = tf.cast(query_masks, tf.bool)
        query_masks = tf.tile(tf.reshape(query_masks, [-1, query_len]), [num_heads, 1])  # (h*N, T_q)
        outputs = tf.reshape(outputs, [-1, key_len])  # (h*N*T_q, T_k)
        paddings = tf.zeros_like(outputs, dtype=tf.float32)  # (h*N*T_q, T_k)
        outputs = tf.where(tf.reshape(query_masks, [-1]), outputs,
                           paddings)  # tf.where((h*N*T_q), (h*N*T_q, T_k), (h*N*T_q, T_k)) => (h*N*T_q, T_k)
        outputs = tf.reshape(outputs, [-1, query_len, key_len])  # (h*N, T_q, T_k)

        # Weighted sum (h*N, T_q, T_k) * (h*N, T_k, C/h)
        outputs = tf.matmul(outputs, V_)  # ( h*N, T_q, C/h)

        # Restore shape
        outputs = tf.concat(tf.split(outputs, num_heads, axis=0), axis=2)  # (N, T_q, C)
        outputs_shape = tf.concat([tf.shape(queries)[:-2], tf.shape(outputs)[-2:]], axis=0)
        outputs = tf.reshape(outputs, outputs_shape)

        if res_connect:
            outputs = outputs+queries
        if final_norm:
            outputs = layers.layer_norm(outputs, begin_norm_axis=-1, begin_params_axis=-1)

        att_vec = tf.reshape(att_vec, [-1, key_len])
        att_vec = tf.where(tf.reshape(query_masks, [-1]), att_vec,
                           tf.zeros_like(att_vec))
        att_vec = tf.reshape(att_vec, [-1, query_len, key_len])
        att_vec = tf.reduce_mean(tf.split(att_vec, num_heads, axis=0), axis=0) # N, T_q, T_k
        
        outputs_shape = tf.concat([tf.shape(queries)[:-2], tf.shape(att_vec)[-2:]], axis=0)
        att_vec = tf.reshape(att_vec, outputs_shape)

    return outputs, att_vec
