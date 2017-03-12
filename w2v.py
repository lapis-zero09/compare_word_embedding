#!/usr/bin/env python
# -*- coding: utf-8 -*-

import gensim
from gensim.models import word2vec
import logging


logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
sentences = word2vec.LineSentence('all_sentences.dat')
model = word2vec.Word2Vec(sentences, sg=1, hs=0, negative=0, size=100, min_count=10, window=10, iter=10)
model.save("./model/SG.model")


model = word2vec.Word2Vec(sentences, sg=1, hs=1, negative=0, size=100, min_count=10, window=10, iter=10)
model.save("./model/SG_with_hs.model")


model = word2vec.Word2Vec(sentences, sg=1, hs=0, negative=15, size=100, min_count=10, window=10, iter=10)
model.save("./model/SG_with_ns15.model")


model = word2vec.Word2Vec(sentences, sg=1, hs=1, negative=15, size=100, min_count=10, window=10, iter=10)
model.save("./model/SG_with_hs_ns15.model")


model = word2vec.Word2Vec(sentences, hs=0, negative=0, size=100, min_count=10, window=10, iter=10)
model.save("./model/CBOW.model")


model = word2vec.Word2Vec(sentences, hs=1, negative=0, size=100, min_count=10, window=10, iter=10)
model.save("./model/CBOW_with_hs.model")


model = word2vec.Word2Vec(sentences, hs=0, negative=15, size=100, min_count=10, window=10, iter=10)
model.save("./model/CBOW_with_ns15.model")


model = word2vec.Word2Vec(sentences, hs=1, negative=15, size=100, min_count=10, window=10, iter=10)
model.save("./model/CBOW_with_hs_ns15.model")
