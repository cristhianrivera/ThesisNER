#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue Aug  8 19:27:41 2017

@author: Cristhian
"""
import tensorflow as tf
import numpy as np
#def train(args):
    

train_inp, train_out = get_train_data()
test_a_inp, test_a_out = get_test_a_data()
test_b_inp, test_b_out = get_test_b_data()
model = Model(args)
maximum = 0
#with tf.Session() as sess:
sess = tf.Session()
sess.run(tf.global_variables_initializer())
#sess.run(tf.initialize_all_variables())

saver = tf.train.Saver()
if args.restore is not None:
    saver.restore(sess, 'model.ckpt')
    print("model restored")
for e in range(args.epoch):
    for ptr in range(0, len(train_inp), args.batch_size):
        
        sess.run(model.train_op, {model.input_data: train_inp[ptr:ptr + args.batch_size],
                                  model.output_data: train_out[ptr:ptr + args.batch_size]})
    if e % 10 == 0:
        save_path = saver.save(sess, "model.ckpt")
        print("model saved in file: %s" % save_path)
    pred, length = sess.run([model.prediction, model.length], {model.input_data: test_a_inp,
                                                               model.output_data: test_a_out})
    print("epoch %d:" % e)
    print('test_a score:')
    m = f1(args, pred, test_a_out, length)
    if m > maximum:
        maximum = m
        save_path = saver.save(sess, "model_max.ckpt")
        print("max model saved in file: %s" % save_path)
        pred, length = sess.run([model.prediction, model.length], {model.input_data: test_b_inp,
                                                                   model.output_data: test_b_out})
        print("test_b score:")
        f1(args, pred, test_b_out, length)
#------------------------------------------------------------------------
        
train_inp, train_out = get_train_data()
test_a_inp, test_a_out = get_test_a_data()
test_b_inp, test_b_out = get_test_b_data()
model = Model(args)
maximum = 0
sess = tf.Session()
sess.run(tf.global_variables_initializer())
sess.run(model.train_op,{model.input_data:train_inp[0:2],
                        model.output_data:train_out[0:2]})

saver = tf.train.Saver()  
save_path = saver.save(sess, "model.ckpt")


pred, length = sess.run([model.prediction, model.length], {model.input_data: test_a_inp[0:2],
                                                           model.output_data: test_a_out[0:2]})

pred, length = sess.run([model.prediction, model.length], {model.input_data: train_inp[0:2],
                                                           model.output_data: train_out[0:2]})

sess.close()    

ta = np.zeros([2,19,111])
tb = np.shape(test_a_inp[0:2])
np.stack([ta,tb])

#------------------------------------------------------------------------


class arg:
    def __init__(self):
        self.word_dim=111
        self.sentence_length = 30
        self.class_size = 5
        self.rnn_size = 256
        self.num_layers = 1
        self.batch_size = 2
        self.epoch = 1
        self.restore = None
        
args = arg()