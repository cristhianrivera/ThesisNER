#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:34:08 2017

@author: Cristhian
"""
import os

os.chdir('embeddings')
from get_conll_embeddings import *
from wordvec_model import *

os.chdir('../data')
from resize_input import *

os.chdir('/home/IAIS/cjimenezri/ner-lstm/ner')
os.getcwd()

#General parameters

max_trim_size = 30

#Resize input

class arg:
    def __init__(self):
        self.input  = 'conll2003/eng.train'
        self.output = 'conll2003/eng_padded_train.txt'
        self.trim = max_trim_size

args = arg()
remove_crap(args.input)
modify_data_size(args.output, args.trim)

os.chdir('/Users/Cristhian/Documents/OneDrive/Documentos/Personal/MSc/Thesis/Fraunhofer/ner-lstm/embeddings')

#Get Conll embeddings
class arg:
    def __init__(self):
        self.train = '../conll2003/eng_padded_train.txt'
        self.test_a = '../conll2003/eng_padded_testa.txt'
        self.test_b = '../conll2003/eng_padded_testb.txt'
        self.sentence_length = -1
        self.use_model = 'wordvec_model_300.pkl'
        self.model_dim = 300

args = arg()
trained_model = pkl.load(open(args.use_model, 'rb'))
get_input(trained_model, args.model_dim, args.train, 'train_embed.pkl', 'train_tag.pkl',
          sentence_length=args.sentence_length)
get_input(trained_model, args.model_dim, args.test_a, 'test_a_embed.pkl', 'test_a_tag.pkl',
          sentence_length=args.sentence_length)
get_input(trained_model, args.model_dim, args.test_b, 'test_b_embed.pkl', 'test_b_tag.pkl',
          sentence_length=args.sentence_length)


#Model
os.chdir('/Users/Cristhian/Documents/ThesisNER')
from model_new import *

class arg:
    def __init__(self):
        self.word_dim=311#300 + POS + Chunk + Capital
        self.sentence_length = max_trim_size #decided by me
        self.class_size = 5 #Conll2003
        self.rnn_size = 256
        self.num_layers = 1
        self.batch_size = 128
        self.epoch = 5
        self.restore = '.'
        
args = arg()
train(args)

predict(args)