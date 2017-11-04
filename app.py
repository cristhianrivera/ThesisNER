#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Nov  4 14:34:08 2017

@author: Cristhian
"""
import os
os.getcwd()
os.chdir('C:\\Python33')
from model_new import *

#Get Conll embeddings
class arg:
    def __init__(self):
        self.train = '../conll2003/eng_padded.train'
        self.test_a = '../conll2003/eng_padded.testa'
        self.test_b = '../conll2003/eng_padded.testb'
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







"""
parser = argparse.ArgumentParser()
parser.add_argument('--word_dim', type=int, help='dimension of word vector', required=True)
parser.add_argument('--sentence_length', type=int, help='max sentence length', required=True)
parser.add_argument('--class_size', type=int, help='number of classes', required=True)
parser.add_argument('--rnn_size', type=int, default=256, help='hidden dimension of rnn')
parser.add_argument('--num_layers', type=int, default=2, help='number of layers in rnn')
parser.add_argument('--batch_size', type=int, default=128, help='batch size of training')
parser.add_argument('--epoch', type=int, default=50, help='number of epochs')
parser.add_argument('--restore', type=str, default=None, help="path of saved model")
train(parser.parse_args())
"""


class arg:
    def __init__(self):
        self.word_dim=311#300 + POS + Chunk + Capial
        self.sentence_length = 30 #decided by me
        self.class_size = 5 #Conll2003
        self.rnn_size = 256
        self.num_layers = 1
        self.batch_size = 128
        self.epoch = 5
        self.restore = None
        
args = arg()
train(args)