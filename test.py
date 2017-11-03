#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 25 10:57:24 2017

@author: Cristhian
"""
import tensorflow as tf
import numpy as np
import argparse

batch_size = 32
max_sen_len = 30
word_dim = 100
rnn_size = 200
n_layers = 1


x = tf.placeholder(tf.float32, [None, max_sen_len, word_dim])

# Prepare data shape to match `rnn` function requirements
# Current data input shape: (batch_size, n_steps, n_input)
# Required shape: 'n_steps' tensors list of shape (batch_size, n_input)

# Unstack to get a list of 'n_steps' tensors of shape (batch_size, n_input)
x_reshape = tf.unstack(x, max_sen_len, 1)

# Define a lstm cell with tensorflow
lstm_cell = tf.contrib.rnn.LSTMCell(rnn_size, forget_bias=1.0)

#simple RNN
# Get lstm cell output
#outputs, states = tf.contrib.rnn.static_rnn(lstm_cell, x_reshape, dtype=tf.float32)

#MultiRNN
#Stack LSTM
rnn_cell = tf.contrib.rnn.MultiRNNCell([lstm_cell for _ in range(n_layers)])    
    

rnn_outputs, rnn_output_states = tf.nn.dynamic_rnn(rnn_cell, x, dtype=tf.float32,time_major = False)

#------------------------------------------------------------------------------
parser = argparse.ArgumentParser()
parser.add_argument('--word_dim', type=int, help='dimension of word vector', required=True)
parser.add_argument('--sentence_length', type=int, help='max sentence length', required=True)
parser.add_argument('--class_size', type=int, help='number of classes', required=True)
parser.add_argument('--rnn_size', type=int, default=256, help='hidden dimension of rnn')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in rnn')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--restore', type=str, default=None, help="path of saved model")


class arg:
    def __init__(self):
        self.word_dim=100
        self.sentence_length = 30
        self.class_size = 5
        self.rnn_size = 256
        self.num_layers = 1 #2
        self.batch_size = 128
        self.epoch = 50
        self.restore = None
      
args = arg()

def lstm_rnn_cell(num_units, dropout):
    _cell = tf.nn.rnn_cell.LSTMCell(num_units,state_is_tuple = True)
    _cell = tf.contrib.rnn.DropoutWrapper(_cell, output_keep_prob=dropout)
    return _cell


  

#1st layer
with tf.variable_scope("layer_1"):
    input_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.word_dim])
    output_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.class_size]) 
    fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_rnn_cell(args.rnn_size, dropout = 0.5) for _ in range(args.num_layers)], 
                                           state_is_tuple = True)
    bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_rnn_cell(args.rnn_size, dropout = 0.5) for _ in range(args.num_layers)], 
                                           state_is_tuple = True)
    words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(input_data), axis = 2))
    length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32) #length of the sentence
    outputs, output_states = tf.nn.bidirectional_dynamic_rnn(
                                                fw_cell, 
                                                bw_cell,
                                                input_data,
                                                dtype = tf.float32, 
                                                sequence_length = length,
                                                parallel_iterations = 64)  

output = tf.concat(outputs,2)
#2nd layer
with tf.variable_scope("layer_2"):
    input_data_2 = tf.placeholder(tf.float32, [None , args.sentence_length , 2 * args.rnn_size])
    fw_cell_2 = tf.nn.rnn_cell.MultiRNNCell([lstm_rnn_cell(args.rnn_size, dropout = 0.5) for _ in range(args.num_layers)],
                                     state_is_tuple = True)
    bw_cell_2 = tf.nn.rnn_cell.MultiRNNCell([lstm_rnn_cell(args.rnn_size, dropout = 0.5) for _ in range(args.num_layers)],
                                     state_is_tuple = True)
    outputs_2, output_states_2 = tf.nn.bidirectional_dynamic_rnn(
                                            fw_cell_2, 
                                            bw_cell_2,
                                            output,
                                            dtype = tf.float32, 
                                            sequence_length = length,
                                            parallel_iterations = 64)  

output_end = tf.concat(outputs_2,2)
weight, bias = weight_and_bias(2 * args.rnn_size, args.class_size)
output = tf.reshape(output, [-1, 2 * args.rnn_size])
prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
prediction = tf.reshape(prediction, [-1, args.sentence_length, args.class_size])
loss = cost()
optimizer = tf.train.AdamOptimizer(0.003)
tvars = tf.trainable_variables()
grads, _ = tf.clip_by_global_norm(tf.gradients(loss, tvars), 10)
train_op = optimizer.apply_gradients(zip(grads, tvars))

        
        
        
        
class Model:
    def __init__(self, args):
        self.args = args
        self.input_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.word_dim])
        self.output_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.class_size])
        fw_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, state_is_tuple=True)
        fw_cell = tf.nn.rnn_cell.DropoutWrapper(fw_cell, output_keep_prob=0.5)
        bw_cell = tf.nn.rnn_cell.LSTMCell(args.rnn_size, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.DropoutWrapper(bw_cell, output_keep_prob=0.5)
        fw_cell = tf.nn.rnn_cell.MultiRNNCell([fw_cell] * args.num_layers, state_is_tuple=True)
        bw_cell = tf.nn.rnn_cell.MultiRNNCell([bw_cell] * args.num_layers, state_is_tuple=True)
        words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(self.input_data), reduction_indices=2))
        self.length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32)
        #output, _, _ = tf.nn.bidirectional_rnn(fw_cell, bw_cell,
        #                                       tf.unpack(tf.transpose(self.input_data, perm=[1, 0, 2])),
        #                                       dtype=tf.float32, sequence_length=self.length)
        
        tf.nn.bidirectional_dynamic_rnn(fw_cell, 
                                        bw_cell,
                                        tf.unstack(tf.transpose(self.input_data, perm=[1, 0, 2])),
                                        dtype=tf.float32, 
                                        sequence_length=self.length)
        
        weight, bias = self.weight_and_bias(2 * args.rnn_size, args.class_size)
        output = tf.reshape(tf.transpose(tf.pack(output), perm=[1, 0, 2]), [-1, 2 * args.rnn_size])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        self.prediction = tf.reshape(prediction, [-1, args.sentence_length, args.class_size])
        self.loss = self.cost()
        optimizer = tf.train.AdamOptimizer(0.003)
        tvars = tf.trainable_variables()
        grads, _ = tf.clip_by_global_norm(tf.gradients(self.loss, tvars), 10)
        self.train_op = optimizer.apply_gradients(zip(grads, tvars))

    def cost(self):
        cross_entropy = self.output_data * tf.log(self.prediction)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        cross_entropy /= tf.cast(self.length, tf.float32)
        return tf.reduce_mean(cross_entropy)

    @staticmethod
    def weight_and_bias(in_size, out_size):
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)
        bias = tf.constant(0.1, shape=[out_size])
        return tf.Variable(weight), tf.Variable(bias)