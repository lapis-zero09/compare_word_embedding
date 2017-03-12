#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import logging
import itertools
import numpy
import gensim
from gensim import utils, matutils
import glove


def make_dict():
    TOKEN_LIMIT = None
    UNIQUE_WORDS = None
    in_file = gensim.models.word2vec.LineSentence('./all_sentences.dat')

    sentences = lambda: itertools.islice(in_file, None)

    logger.info("dictionary creating")
    id2word = gensim.corpora.Dictionary(in_file, prune_at=UNIQUE_WORDS)
    id2word.filter_extremes(keep_n=TOKEN_LIMIT)

    word2id = dict((v, k) for k, v in id2word.items())
    # utils.pickle(word2id, './tmp/word2id.glove')
    id2word = gensim.utils.revdict(word2id)

    corpus = lambda: ([word for word in sentence if word in word2id] for sentence in sentences())
    return corpus, word2id, id2word

def train(word2id, id2word, corpus, win, dim):
    cooccur = glove.Corpus(dictionary=word2id)
    cooccur.fit(corpus(), window=win)

    logger.info("glove model creating")
    logger.info('Dict size: %s' % len(cooccur.dictionary))
    logger.info('Collocations: %s' % cooccur.matrix.nnz)
    model = glove.Glove(no_components=dim, learning_rate=0.05)
    model.fit(cooccur.matrix, epochs=10, no_threads=5, verbose=True)
    model.add_dictionary(cooccur.dictionary)
    model.word2id = dict((utils.to_unicode(w), id) for w, id in model.dictionary.items())
    model.id2word = gensim.utils.revdict(model.word2id)
    utils.pickle(model, './model/glove.model')

logger = logging.getLogger("glove")

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

    logger.info("glove corpus matrix creating")
    corpus, word2id, id2word = make_dict()

    train(word2id, id2word, corpus, win=10, dim=100)
