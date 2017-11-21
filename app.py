#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:34:08 2017

@author: Cristhian
"""
import os
os.chdir('/Users/Cristhian/Documents/ThesisNER')
os.chdir('embeddings')
from get_conll_embeddings import *
from wordvec_model import *

os.chdir('../data')
from resize_input import *

#os.chdir('/home/IAIS/cjimenezri/ner-lstm/ner')
os.chdir('/Users/Cristhian/Documents/OneDrive/Documentos/Personal/MSc/Thesis/Fraunhofer/ner-lstm/')
os.getcwd()

#General parameters

max_trim_size = 30

#Resize input

class arg:
    def __init__(self):
        self.input  = 'conll2003/deu.testb'
        self.output = 'conll2003/deu_padded_testb.txt'
        self.trim = max_trim_size

args = arg()
remove_crap(args.input)
modify_data_size(args.output, args.trim)

os.chdir('/Users/Cristhian/Documents/OneDrive/Documentos/Personal/MSc/Thesis/Fraunhofer/ner-lstm/embeddings')

#Get Conll embeddings
class arg:
    def __init__(self):
        self.train = '../conll2003/deu_padded_train.txt'
        self.test_a = '../conll2003/deu_padded_testa.txt'
        self.test_b = '../conll2003/deu_padded_testb.txt'
        self.sentence_length = -1
        self.use_model = 'deu_wordvec_model_300.pkl'
        self.model_dim = 300

args = arg()
trained_model = pkl.load(open(args.use_model, 'rb'))
get_input_deu(trained_model, args.model_dim, args.train, 'deu_train_embed.pkl', 'deu_train_tag.pkl',
          sentence_length=args.sentence_length)
get_input_deu(trained_model, args.model_dim, args.test_a, 'deu_test_a_embed.pkl', 'deu_test_a_tag.pkl',
          sentence_length=args.sentence_length)
get_input_deu(trained_model, args.model_dim, args.test_b, 'deu_test_b_embed.pkl', 'deu_test_b_tag.pkl',
          sentence_length=args.sentence_length)




#Model
import os
import numpy as np
os.chdir('/Users/Cristhian/Documents/ThesisNER')
from model_new import *


class arg:
    def __init__(self):
        self.word_dim=301#300 + POS + Capital
        self.sentence_length = 30 #decided by me
        self.class_size = 5 #Conll2003
        self.rnn_size = 256
        self.num_layers = 1
        self.batch_size = 128
        self.epoch = 100
        self.restore = None
        self.model_name = 'German_Original'
        
args = arg()
train(args)

predict(args)