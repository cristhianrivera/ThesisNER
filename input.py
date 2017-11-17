from __future__ import print_function
import pickle

setDir = '/home/IAIS/cjimenezri/ner-lstm/ner/embeddings/'

def get_train_data():
    emb = pickle.load(open(setDir + 'train_embed.pkl', 'rb'))
    tag = pickle.load(open(setDir + 'train_tag.pkl', 'rb'))
    print('train data loaded')
    return emb, tag


def get_test_a_data():
    emb = pickle.load(open(setDir + 'test_a_embed.pkl', 'rb'))
    tag = pickle.load(open(setDir + 'test_a_tag.pkl', 'rb'))
    print('test_a data loaded')
    return emb, tag


def get_test_b_data():
    emb = pickle.load(open(setDir + 'test_b_embed.pkl', 'rb'))
    tag = pickle.load(open(setDir + 'test_b_tag.pkl', 'rb'))
    print('test_b data loaded')
    return emb, tag



