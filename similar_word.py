#!/usr/bin/env python
# -*- coding: utf-8 -*-

import itertools
import numpy
import scipy.sparse

import gensim
from gensim import utils, matutils

def most_similar_dist(model, positive=[], negative=[], topn=10):
    if isinstance(positive, str) and not negative:
        positive = [positive]

    positive = [
        (word, 1.0) if isinstance(word, (str, numpy.ndarray)) else word
        for word in positive]
    negative = [
        (word, -1.0) if isinstance(word, (str, numpy.ndarray)) else word
        for word in negative]

    all_words, mean = set(), []
    for word, weight in positive + negative:
        if isinstance(word, numpy.ndarray):
            mean.append(weight * word)
        elif word in model.word2id:
            word_index = model.word2id[word]
            mean.append(weight * model.word_vectors[word_index])
            all_words.add(word_index)
        else:
            raise KeyError("word '%s' not in vocabulary" % word)
    if not mean:
        raise ValueError("cannot compute similarity with no input")
    if scipy.sparse.issparse(model.word_vectors):
        mean = scipy.sparse.vstack(mean)
    else:
        mean = numpy.array(mean)
    mean = matutils.unitvec(mean.mean(axis=0)).astype(model.word_vectors.dtype)

    dists = model.word_vectors.dot(mean.T).flatten()
    if not topn:
        return dists
    best = numpy.argsort(dists)[::-1][:topn + len(all_words)]

    result = [(model.id2word[sim], float(dists[sim])) for sim in best if sim not in all_words]

    return result[:topn]
