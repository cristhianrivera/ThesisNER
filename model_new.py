from __future__ import print_function
from tensorflow.python.saved_model import builder as saved_model_builder
import tensorflow as tf
import numpy as np
import argparse
import pickle


def get_train_data():
    emb = pickle.load(open('/Users/Cristhian/Documents/OneDrive/Documentos/Personal/MSc/Thesis/Fraunhofer/ner-lstm/embeddings/train_embed.pkl', 'rb'))
    tag = pickle.load(open('/Users/Cristhian/Documents/OneDrive/Documentos/Personal/MSc/Thesis/Fraunhofer/ner-lstm/embeddings/train_tag.pkl', 'rb'))
    print('train data loaded')
    return emb, tag


def get_test_a_data():
    emb = pickle.load(open('/Users/Cristhian/Documents/OneDrive/Documentos/Personal/MSc/Thesis/Fraunhofer/ner-lstm/embeddings/test_a_embed.pkl', 'rb'))
    tag = pickle.load(open('/Users/Cristhian/Documents/OneDrive/Documentos/Personal/MSc/Thesis/Fraunhofer/ner-lstm/embeddings/test_a_tag.pkl', 'rb'))
    print('test_a data loaded')
    return emb, tag


def get_test_b_data():
    emb = pickle.load(open('/Users/Cristhian/Documents/OneDrive/Documentos/Personal/MSc/Thesis/Fraunhofer/ner-lstm/embeddings/test_b_embed.pkl', 'rb'))
    tag = pickle.load(open('/Users/Cristhian/Documents/OneDrive/Documentos/Personal/MSc/Thesis/Fraunhofer/ner-lstm/embeddings/test_b_tag.pkl', 'rb'))
    print('test_b data loaded')
    return emb, tag


def lstm_rnn_cell(num_units, dropout):
    _cell = tf.nn.rnn_cell.LSTMCell(num_units,state_is_tuple = True)
    _cell = tf.contrib.rnn.DropoutWrapper(_cell, output_keep_prob = dropout)
    return _cell


class Model:
    def __init__(self, args):
        self.args = args
        self.input_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.word_dim], name = "input")
        self.output_data = tf.placeholder(tf.float32, [None, args.sentence_length, args.class_size], name = "output")
        self.dropout = tf.placeholder(tf.float32,name = "dropout")
        with tf.variable_scope("layer_1"):
            fw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_rnn_cell(args.rnn_size, dropout = self.dropout) for _ in range(args.num_layers)], 
                                                   state_is_tuple = True)
            bw_cell = tf.nn.rnn_cell.MultiRNNCell([lstm_rnn_cell(args.rnn_size, dropout = self.dropout ) for _ in range(args.num_layers)], 
                                                   state_is_tuple = True)
                   
            words_used_in_sent = tf.sign(tf.reduce_max(tf.abs(self.input_data), axis = 2))
            self.length = tf.cast(tf.reduce_sum(words_used_in_sent, reduction_indices=1), tf.int32) #length of the sentence
            outputs, output_states = tf.nn.bidirectional_dynamic_rnn(fw_cell, 
                                                            bw_cell,
                                                            self.input_data,
                                                            dtype=tf.float32, 
                                                            sequence_length = self.length,
                                                            parallel_iterations = 128)  
        output = tf.concat(outputs,2)
        with tf.variable_scope("layer_2"):
            fw_cell_2 = tf.nn.rnn_cell.MultiRNNCell([lstm_rnn_cell(args.rnn_size, dropout = self.dropout) for _ in range(args.num_layers)], 
                                                   state_is_tuple = True)
            bw_cell_2 = tf.nn.rnn_cell.MultiRNNCell([lstm_rnn_cell(args.rnn_size, dropout = self.dropout) for _ in range(args.num_layers)], 
                                                   state_is_tuple = True)
            outputs_2, output_states_2 = tf.nn.bidirectional_dynamic_rnn(
                                            fw_cell_2, 
                                            bw_cell_2,
                                            output,
                                            dtype = tf.float32, 
                                            sequence_length = self.length,
                                            parallel_iterations = 128)  
        output_end = tf.concat(outputs_2,2)
                   
        
        weight, bias = self.weight_and_bias(2 * args.rnn_size, args.class_size)
        output = tf.reshape(output_end, [-1, 2 * args.rnn_size])
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        #output_java = tf.identity(prediction, name = "output_java")
        self.prediction = tf.reshape(prediction, [-1, args.sentence_length, args.class_size])
        output_java = tf.identity(self.prediction, name = "output_java")
        self.loss = self.cost()
        #optimizer = tf.train.AdamOptimizer(0.003)#RMSProp optimizer
        optimizer = tf.train.RMSPropOptimizer(0.003)
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
        weight = tf.truncated_normal([in_size, out_size], stddev=0.01)#Xavier_initializer 
        #weight = tf.get_variable("weight", shape=[in_size, out_size], initializer=tf.contrib.layers.xavier_initializer())
        #initializer = tf.contrib.layers.xavier_initializer()
        #weight = tf.Variable(initializer([in_size,out_size]))
        #initializer = tf.contrib.layers.xavier_initializer()
        bias = tf.constant(0.1, shape=[out_size])
        #return tf.Variable(initializer([in_size,out_size])), tf.Variable(bias)
        return tf.Variable(weight), tf.Variable(bias)


def f1(args, prediction, target, length):
    tp = np.array([0] * (args.class_size + 1))
    fp = np.array([0] * (args.class_size + 1))
    fn = np.array([0] * (args.class_size + 1))
    target = np.argmax(target, 2)
    prediction = np.argmax(prediction, 2)
    for i in range(len(target)):
        for j in range(length[i]):
            if target[i, j] == prediction[i, j]:
                tp[target[i, j]] += 1
            else:
                fp[target[i, j]] += 1
                fn[prediction[i, j]] += 1
    unnamed_entity = args.class_size - 1
    for i in range(args.class_size):
        if i != unnamed_entity:
            tp[args.class_size] += tp[i]
            fp[args.class_size] += fp[i]
            fn[args.class_size] += fn[i]
    precision = []
    recall = []
    fscore = []
    for i in range(args.class_size + 1):
        precision.append(tp[i] * 1.0 / (tp[i] + fp[i]))
        recall.append(tp[i] * 1.0 / (tp[i] + fn[i]))
        fscore.append(2.0 * precision[i] * recall[i] / (precision[i] + recall[i]))
    print ("precision ", precision)
    print ("recall ", recall)
    print ("f1 score ", fscore)
    return fscore[args.class_size]


def train(args):
    train_inp, train_out = get_train_data()
    test_a_inp, test_a_out = get_test_a_data()
    test_b_inp, test_b_out = get_test_b_data()
    model = Model(args)
    maximum = 0
    #builder = tf.saved_model.builder.SavedModelBuilder("./model")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.initialize_all_variables())
        
        saver = tf.train.Saver()
        if args.restore is not None:
            saver.restore(sess, 'model.ckpt')
            print("model restored")
            
        ff = open('loss.txt', 'w')
        for e in range(args.epoch):
            for ptr in range(0, len(train_inp), args.batch_size):
                sess.run(model.train_op, {model.input_data: train_inp[ptr:ptr + args.batch_size],
                                          model.output_data: train_out[ptr:ptr + args.batch_size],
                                          model.dropout: 0.5})
            if e % 10 == 0:
                save_path = saver.save(sess, "model.ckpt")
                print("model saved in file: %s" % save_path)
            pred, length , loss = sess.run([model.prediction, model.length, model.loss], {model.input_data: test_a_inp,
                                                                       model.output_data: test_a_out,
                                                                       model.dropout: 1.0 })
            
            ff.writelines("%d \t %f" % (e ,loss))
            
            print("loss = %s" % loss)
            print("epoch %d:" % e)
            print('test_a score:')
            m = f1(args, pred, test_a_out, length)
            if m > maximum:
                maximum = m
                save_path = saver.save(sess, "model_max.ckpt")
                print("max model saved in file: %s" % save_path)
                pred, length = sess.run([model.prediction, model.length], {model.input_data: test_b_inp,
                                                                           model.output_data: test_b_out,
                                                                           model.dropout: 1.0 })
                print("test_b score:")
                f1(args, pred, test_b_out, length)
        ff.close()  
        #builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
        #builder.save(True)



def predict(args):
    test_a_inp, test_a_out = get_test_a_data()
    test_b_inp, test_b_out = get_test_b_data()
    model = Model(args)
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.initialize_all_variables())
        
        saver = tf.train.Saver()
        saver.restore(sess, 'model_max.ckpt')
        print("model restored")
        pred, length , loss = sess.run([model.prediction, model.length, model.loss], {model.input_data: test_a_inp,
                                                                       model.output_data: test_a_out,
                                                                       model.dropout: 1.0 })
        m = f1(args, pred, test_a_out, length)

