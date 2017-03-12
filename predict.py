#!/usr/bin/env python
# -*- coding: utf-8 -*-

from gensim.models.word2vec import Word2Vec
from gensim import utils
from glove import Glove
from similar_word import most_similar_dist
from collections import defaultdict


# def calculate_score(result, score={}):
#     if not result is None:
#         s = sum([v for k, v in result])
#         for i, (k, v) in enumerate(sorted(result, key=lambda x: x[1], reverse=True)):
#             # method of calculating score
#         return score

def load_model(model, topn, positive=[], negative=[]):
    if model == 'glove' or model == 'ppmi' or model == 'svd':
        model = utils.unpickle('./model/{}.model'.format(model))
        return most_similar_dist(model, positive=positive, negative=negative, topn=topn)
    else:
        model = Word2Vec.load('./model/{}.model'.format(model))
        return model.most_similar(positive=positive, negative=negative, topn=topn)

def predict(topn, positive=[], negative=[]):
    # score = defaultdict(float)
    model = ['CBOW_with_hs', 'CBOW_with_ns15', 'CBOW_with_hs_ns15',
             'SG_with_hs', 'SG_with_ns15', 'SG_with_hs_ns15',
             'glove', 'ppmi', 'svd']

    for m in model:
        print(m)
        result = load_model(model=m, positive=positive, negative=negative, topn=topn)
        print(result)

    # return score


if __name__ == '__main__':
    ret = predict(positive=['iphone'], negative=[], topn=10)
