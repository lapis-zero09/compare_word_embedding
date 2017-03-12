#!/usr/bin/env python
# -*- coding: utf-8 -*-


import os
import sys
import logging
import itertools
import numpy
import gensim
from gensim import utils, matutils


WINDOW = 10
TOKEN_LIMIT = None
DYNAMIC_WINDOW = True
NEGATIVE = 10
UNIQUE_WORDS = None

logger = logging.getLogger("ppmi")

import pyximport; pyximport.install(setup_args={'include_dirs': numpy.get_include()})
from cooccur_matrix import get_cooccur


def raw2ppmi(cooccur, word2id, k_shift=1.0):

    logger.info("computing PPMI on co-occurence counts")


    marginal_word = cooccur.sum(axis=1)
    marginal_context = cooccur.sum(axis=0)
    cooccur /= marginal_word[:, None]  # #(w, c) / #w
    cooccur /= marginal_context  # #(w, c) / (#w * #c)
    cooccur *= marginal_word.sum()  # #(w, c) * D / (#w * #c)
    numpy.log(cooccur, out=cooccur)  # PMI = log(#(w, c) * D / (#w * #c))

    logger.info("shifting PMI scores by log(k) with k=%s" % (k_shift, ))
    cooccur -= numpy.log(k_shift)  # shifted PMI = log(#(w, c) * D / (#w * #c)) - log(k)

    logger.info("clipping PMI scores to be non-negative PPMI")
    cooccur.clip(0.0, out=cooccur)  # SPPMI = max(0, log(#(w, c) * D / (#w * #c)) - log(k))

    logger.info("normalizing PPMI word vectors to unit length")
    for i, vec in enumerate(cooccur):
        cooccur[i] = matutils.unitvec(vec)

    return matutils.Dense2Corpus(cooccur, documents_columns=False)


class PpmiModel(object):
    def __init__(self, corpus):

        self.word_vectors = matutils.corpus2csc(corpus).T


if __name__ == "__main__":
    logging.basicConfig(format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s', level=logging.INFO)

    from ppmi import PpmiModel

    in_file = gensim.models.word2vec.LineSentence('./all_sentences.dat')

    ##
    sentences = lambda: itertools.islice(in_file, None)

    logger.info("dictionary creating")
    id2word = gensim.corpora.Dictionary(in_file, prune_at=UNIQUE_WORDS)
    id2word.filter_extremes(keep_n=TOKEN_LIMIT)

    word2id = dict((v, k) for k, v in id2word.items())
    utils.pickle(word2id, './tmp/word2id')

    id2word = gensim.utils.revdict(word2id)

    ## filter sentences to contain only the dictionary words
    corpus = lambda: ([word for word in sentence if word in word2id] for sentence in sentences())


    logger.info("PMI matrix creating")

    logger.info("raw cooccurrence matrix creating")
    raw = get_cooccur(corpus(), word2id, window=WINDOW, dynamic_window=DYNAMIC_WINDOW)
    numpy.save('./tmp/cooccur.npy', raw)
    # store the SPPMI matrix in sparse Matrix Market format on disk
    gensim.corpora.MmCorpus.serialize('./tmp/pmi_matrix.mm', raw2ppmi(raw, word2id, k_shift=NEGATIVE or 1))
    del raw


    logger.info("PMI model creating")
    model = PpmiModel(gensim.corpora.MmCorpus('./tmp/pmi_matrix.mm'))
    model.word2id = word2id
    model.id2word = id2word
    utils.pickle(model, './model/ppmi.model')

    logger.info("finished running pmi")
