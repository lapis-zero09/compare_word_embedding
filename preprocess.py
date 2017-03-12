#!/usr/bin/env python
# -*- coding: utf-8 -*-

import MeCab
import mojimoji
import re
import os
from glob import glob


def open_file(file):
    with open(file, 'r') as f:
        lines = f.readlines()

    ret = ''
    for i, line in enumerate(lines):
        if i > 1:
            ret += line.replace('\n', ' ')

    return ret

def directory_parse(directory):
    ret_dict = {}
    file_list = glob(directory + '/*')
    for f in file_list:
        if os.path.isdir(f):
            ret_dict.update(directory_parse(f))
        elif 'LICENSE' in f or 'README' in f:
            pass
        else:
            ret_dict[f] = open_file(f)
    return ret_dict

class Tokenizer():
    '''
    data to token
    '''

    def __init__(self):
        self.mt = MeCab.Tagger('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

    def fit_on_texts(self, sentence):
        node = self.mt.parseToNode(sentence)
        while node:
            feature = node.feature.split(',')
            if feature[0] != '記号':
                ret = mojimoji.zen_to_han(feature[-3], kana=False).lower()
                if ret != '*' and ret != ' ':
                    yield ret
            node = node.next

    def texts_to_sequences(self, content):
        ret = ''
        for item in self.fit_on_texts(content):
            ret += item
            ret += ' '
        return ret

    def tokenize(self, contents):
        ret_list = []
        for k, content in contents.items():
            ret_list.append(self.texts_to_sequences(content))
        return ret_list


tk = Tokenizer()
contents = directory_parse('./text/')
words = tk.tokenize(contents)

with open('all_sentences.dat', 'w') as f:
    for word in words:
        # print(word)
        f.write(word)
        f.write('\n')
