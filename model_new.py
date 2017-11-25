from __future__ import print_function
from tensorflow.python.saved_model import builder as saved_model_builder
from sklearn.utils import shuffle
import tensorflow as tf
import numpy as np
import argparse
import pickle

# for the embeddings
setDir = '/home/IAIS/cjimenezri/ner-lstm/ner/embeddings/'
#language = 'esp_combined'
#setDir = '/Users/Cristhian/Documents/OneDrive/Documentos/Personal/MSc/Thesis/Fraunhofer/ner-lstm/embeddings/'
max_trim_size = 30
    
def get_train_data(language):
    emb = pickle.load(open(setDir + language + '_train_embed_' + str(max_trim_size) + '.pkl', 'rb'))
    tag = pickle.load(open(setDir + language + '_train_tag_' + str(max_trim_size) + '.pkl', 'rb'))
    print('train '+ language + ' ' + str(max_trim_size) + ' data loaded')
    return emb, tag


def get_test_a_data(language):
    emb = pickle.load(open(setDir + language + '_test_a_embed_' + str(max_trim_size) + '.pkl', 'rb'))
    tag = pickle.load(open(setDir + language + '_test_a_tag_' + str(max_trim_size) + '.pkl', 'rb'))
    print('test_a ' + str(max_trim_size) + ' data loaded')
    return emb, tag


def get_test_b_data(language):
    emb = pickle.load(open(setDir + language + '_test_b_embed_' + str(max_trim_size) + '.pkl', 'rb'))
    tag = pickle.load(open(setDir + language + '_test_b_tag_' + str(max_trim_size) + '.pkl', 'rb'))
    print('test_b ' + str(max_trim_size) + ' data loaded')
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
        self.dropout = tf.placeholder(tf.float32, name = "dropout")
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
        #prediction = tf.nn.tanh( tf.matmul(output, weight) + bias)
        #prediction = tf.nn.softmax(prediction)
        prediction = tf.nn.softmax(tf.matmul(output, weight) + bias)
        self.prediction = tf.reshape(prediction, [-1, args.sentence_length, args.class_size])
        #self.prediction = tf.clip_by_value(prediction,1e-6,1.0)
        output_java = tf.identity(self.prediction, name = "output_java")
        self.loss = self.cost()
        optimizer = tf.train.AdamOptimizer(0.0005)
        gvs = optimizer.compute_gradients(self.loss)
        capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
        self.train_op = optimizer.apply_gradients(capped_gvs)
        

    def cost(self):
        cross_entropy = self.output_data * tf.log(self.prediction+ 1e-6)
        cross_entropy = -tf.reduce_sum(cross_entropy, reduction_indices=2)
        mask = tf.sign(tf.reduce_max(tf.abs(self.output_data), reduction_indices=2))
        cross_entropy *= mask
        cross_entropy = tf.reduce_sum(cross_entropy, reduction_indices=1)
        #cross_entropy /= tf.cast(self.length, tf.float32)
        
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
    #print ("precision ", precision)
    #print ("recall ", recall)
    print ("f1 score ", fscore)
    return fscore[args.class_size],precision[args.class_size],recall[args.class_size]


def train(args):
    train_inp_eng, train_out_eng = get_train_data('eng_combined')
    test_a_inp_eng, test_a_out_eng = get_test_a_data('eng_combined')
    test_b_inp_eng, test_b_out_eng = get_test_b_data('eng_combined')
    
    train_inp_esp, train_out_esp = get_train_data('esp_combined')
    test_a_inp_esp, test_a_out_esp = get_test_a_data('esp_combined')
    test_b_inp_esp, test_b_out_esp = get_test_b_data('esp_combined')
    
    
    train_inp_deu, train_out_deu = get_train_data('deu_combined')
    test_a_inp_deu, test_a_out_deu = get_test_a_data('deu_combined')
    test_b_inp_deu, test_b_out_deu = get_test_b_data('deu_combined')
    """
    train_inp_deu = np.asarray(train_inp_deu)
    train_out_deu = np.asarray(train_out_deu)
    train_inp_eng = np.asarray(train_inp_eng)
    train_out_eng = np.asarray(train_out_eng)
    train_inp_esp = np.asarray(train_inp_esp)
    train_out_esp = np.asarray(train_out_esp)
    
    test_a_inp_deu = np.asarray(test_a_inp_deu)
    test_a_out_deu = np.asarray(test_a_out_deu)
    test_a_inp_eng = np.asarray(test_a_inp_eng)
    test_a_out_eng = np.asarray(test_a_out_eng)
    test_a_inp_esp = np.asarray(test_a_inp_esp)
    test_a_out_esp = np.asarray(test_a_out_esp)
    
    test_b_inp_deu = np.asarray(test_b_inp_deu)
    test_b_out_deu = np.asarray(test_b_out_deu)
    test_b_inp_eng = np.asarray(test_b_inp_eng)
    test_b_out_eng = np.asarray(test_b_out_eng)
    test_b_inp_esp = np.asarray(test_b_inp_esp)
    test_b_out_esp = np.asarray(test_b_out_esp)
    
    train_inp = np.concatenate((train_inp_deu, train_inp_eng, train_inp_esp), axis = 0)
    train_out = np.concatenate((train_out_deu, train_out_eng, train_out_esp), axis = 0)
    
    test_a_inp = np.concatenate((test_a_inp_deu, test_a_inp_eng, test_a_inp_esp), axis = 0)
    test_a_out = np.concatenate((test_a_out_deu, test_a_out_eng, test_a_out_esp), axis = 0)
    
    test_b_inp = np.concatenate((test_b_inp_deu, test_b_inp_eng, test_b_inp_esp), axis = 0)
    test_b_out = np.concatenate((test_b_out_deu, test_b_out_eng, test_b_out_esp), axis = 0)
    """
    train_inp = np.concatenate((train_inp_eng, train_inp_esp, train_inp_deu), axis = 0)
    train_out = np.concatenate((train_out_eng, train_out_esp, train_out_deu), axis = 0)
    
    test_a_inp = np.concatenate((test_a_inp_eng, test_a_inp_esp, test_a_inp_deu), axis = 0)
    test_a_out = np.concatenate((test_a_out_eng, test_a_out_esp, test_a_out_deu), axis = 0)
    
    test_b_inp = np.concatenate((test_b_inp_eng, test_b_inp_esp, test_b_inp_deu), axis = 0)
    test_b_out = np.concatenate((test_b_out_eng, test_b_out_esp, test_b_out_deu), axis = 0)
    
    train_inp, train_out = shuffle(train_inp, train_out)
    test_a_inp, test_a_out = shuffle(test_a_inp, test_a_out)
    test_b_inp, test_b_out = shuffle(test_b_inp, test_b_out)
    
    model = Model(args)
    train_loss = 0
    maximum = 0
    #builder = tf.saved_model.builder.SavedModelBuilder("./model")
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        #sess.run(tf.initialize_all_variables())

        saver = tf.train.Saver()
        if args.restore is not None:
            saver.restore(sess, 'model_' + args.model_name + '.ckpt')
            print("model restored")
            
        ff = open('StatisticalNER/' + args.model_name + '_' + str(args.sentence_length) + '_loss.txt', 'w')
        
        ff1 = open('StatisticalNER/' + args.model_name + '_' + str(args.sentence_length) + '_f1.txt', 'w')
        fprecision = open('StatisticalNER/' + args.model_name + '_' + str(args.sentence_length) + '_precision.txt', 'w')
        frecall = open('StatisticalNER/' + args.model_name + '_' + str(args.sentence_length) + '_recall.txt', 'w')
        
        
        for e in range(args.epoch):   
            print ("Len of train: " + str(len(train_inp)))
            for ptr in range(0, len(train_inp), args.batch_size):

                assert not np.any(np.isnan(train_inp[ptr:ptr + args.batch_size]))
                
                _ , t_loss, predi = sess.run([model.train_op, model.loss, model.prediction], 
                                             {model.input_data: train_inp[ptr:ptr + args.batch_size],
                                          model.output_data: train_out[ptr:ptr + args.batch_size],
                                          model.dropout: 0.5})
                    
                        
            if e % 10 == 0:
                save_path = saver.save(sess, 'model_' + args.model_name + "_" + str(e) + '.ckpt')
                print("model saved in file: %s" % save_path)
                
            pred, length , loss = sess.run([model.prediction, model.length, model.loss], {model.input_data: test_a_inp,
                                                                       model.output_data: test_a_out,
                                                                       model.dropout: 1.0 })
            
            ff.writelines("%d \t %f" % (e ,loss))
                
            print("loss = %.4f" % loss)
            print("epoch %d:" % e)
            print('test_a score:')
                            
            m = f1(args, pred, test_a_out, length)
            
            fprecision.writelines("%d \t %f" % (e ,m[1]))
            frecall.writelines("%d \t %f" % (e ,m[2]))
            ff1.writelines("%d \t %f" % (e ,m[0]))
            
            if m[0] > maximum:
                maximum = m[0]
                save_path = saver.save(sess, 'model_' + args.model_name + '_max.ckpt')
                print("max model saved in file: %s" % save_path)
                pred, length = sess.run([model.prediction, model.length], {model.input_data: test_b_inp,
                                                                           model.output_data: test_b_out,
                                                                           model.dropout: float(1.0) })
                print("test_b score:")
                f1(args, pred, test_b_out, length)

            
        
        ff.close()  

        fprecision.close()
        frecall.close()
        ff1.close()
        
        #builder.add_meta_graph_and_variables(sess, [tf.saved_model.tag_constants.SERVING])
        #builder.save(True)



def predict(args):
    train_inp_eng, train_out_eng = get_train_data('eng_combined')
    test_a_inp_eng, test_a_out_eng = get_test_a_data('eng_combined')
    test_b_inp_eng, test_b_out_eng = get_test_b_data('eng_combined')
    
    train_inp_esp, train_out_esp = get_train_data('esp_combined')
    test_a_inp_esp, test_a_out_esp = get_test_a_data('esp_combined')
    test_b_inp_esp, test_b_out_esp = get_test_b_data('esp_combined')
    
    train_inp_deu, train_out_deu = get_train_data('deu_combined')
    test_a_inp_deu, test_a_out_deu = get_test_a_data('deu_combined')
    test_b_inp_deu, test_b_out_deu = get_test_b_data('deu_combined')

    train_inp = np.concatenate((train_inp_eng, train_inp_esp, train_inp_deu), axis = 0)
    train_out = np.concatenate((train_out_eng, train_out_esp, train_out_deu), axis = 0)
    
    test_a_inp = np.concatenate((test_a_inp_eng, test_a_inp_esp, test_a_inp_deu), axis = 0)
    test_a_out = np.concatenate((test_a_out_eng, test_a_out_esp, test_a_out_deu), axis = 0)
    
    test_b_inp = np.concatenate((test_b_inp_eng, test_b_inp_esp, test_b_inp_deu), axis = 0)
    test_b_out = np.concatenate((test_b_out_eng, test_b_out_esp, test_b_out_deu), axis = 0)
    
    train_inp, train_out = shuffle(train_inp, train_out)
    test_a_inp, test_a_out = shuffle(test_a_inp, test_a_out)
    test_b_inp, test_b_out = shuffle(test_b_inp, test_b_out)

    print ("Len of train: " + str(len(train_inp)))
    print ("Len of test a: " + str(len(test_a_inp)))
    print ("Len of test b: " + str(len(test_b_inp)))
    
    model = Model(args)
    
    with tf.Session() as sess:
       
        saver = tf.train.Saver()
        saver.restore(sess, 'model_All_30_CCA512_max.ckpt')
        print("model restored")
        
        print ('\n --------- test a data ------------')
        pred, length , loss = sess.run([model.prediction, model.length, model.loss], 
                                           {model.input_data: test_a_inp,
                                            model.output_data: test_a_out,
                                           model.dropout: 1 })
        m = f1(args, pred, test_a_out, length)
        
        print ('\n --------- test b data ------------')
        pred, length , loss = sess.run([model.prediction, model.length, model.loss], 
                                           {model.input_data: test_b_inp,
                                            model.output_data: test_b_out,
                                           model.dropout: 1 })
        m = f1(args, pred, test_b_out, length)
        
        
        print ('\n --------- training data ------------')
        for ptr in range(0, len(train_inp), args.batch_size):
            pred, length , loss = sess.run([model.prediction, model.length, model.loss], 
                                           {model.input_data: train_inp[ptr:ptr + args.batch_size],
                                            model.output_data: train_out[ptr:ptr + args.batch_size],
                                           model.dropout: 1 })
            m = f1(args, pred, train_out[ptr:ptr + args.batch_size], length)

