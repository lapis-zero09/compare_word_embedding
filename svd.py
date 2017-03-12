#!/usr/bin/env python
# -*- coding: utf-8 -*-


import logging
import numpy
import gensim
from gensim import utils, matutils


DIM = 100
logger = logging.getLogger("svd")

class SvdModel(object):
    def __init__(self, corpus, id2word, s_exponent=0.0):
        logger.info("make dictionary from corpus")
        dictionary = gensim.corpora.dictionary.Dictionary.from_corpus(corpus, id2word=id2word)

        logger.info("calculating truncated SVD")
        lsi = gensim.models.LsiModel(corpus, id2word=dictionary, num_topics=DIM)
        self.singular_scaled = lsi.projection.s ** s_exponent
        # embeddings = left singular vectors scaled by the (exponentiated) singular values
        self.word_vectors = lsi.projection.u * self.singular_scaled

if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

    from svd import SvdModel

    word2id = utils.unpickle('./tmp/word2id')
    id2word = gensim.utils.revdict(word2id)

    logger.info("SVD model creating")
    corpus = gensim.corpora.MmCorpus('./tmp/pmi_matrix.mm')
    model = SvdModel(corpus, id2word, s_exponent=0.0)
    model.word2id = word2id
    model.id2word = id2word
    utils.pickle(model, './tmp/svd.model')


    logger.info("finished running svd")
