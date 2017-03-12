# compare_word_embedding


## Pre-processing

```
$ mkdir tmp
$ mkdir model
$ python preprocess.py
```

## Train

```
$ python w2v.py
$ python glove_train.py
$ python ppmi.py
$ python svd.py
```

## Evaluate

```
$ python evaluate.py
```

## methods

- Distributional semantic representation
  - Positive Pointwise Mutual Information(PPMI)

  - Singular Value Decomposition(SVD)


- Distributed semantic representation
  - Word2Vec
      - Continuous Bag Of Words(CBOW)
      - Skip-gram(SG)

  - GloVe

## Dependencies

### Python-version

- Python 3.6.0 :: Anaconda 4.3.1 (x86_64)

### MeCab dictionary

- mecab-ipadic-neologd

### Packages

- gensim==0.13.4.1
- glove-python==0.1.0
- mecab-python3==0.7

## how to install glove-python on OS X El Capitan

[glove-python](https://github.com/maciejkula/glove-python)はデフォルトでは動かない．

Macの標準コンパイラがgccではなくclangだからである．

glove-pythonには環境を調査し，それに応じたコンパイラを指定する機能があるがそれが
標準では動作していない．

Githubからcloneする

```
$ git clone https://github.com/maciejkula/glove-python.git
```

setup.pyを編集

define_extensions関数のはじめにset_gcc関数を呼び出すようにする


```

def define_extensions(cythonize=False):
    set_gcc()
    compile_args = ['-fopenmp',
                    '-ffast-math']

```

以下を実行するとインストールできる

```
$ sudo python setup.py install
```

できない場合はgcc周りの環境が整っていないので，gccの4.9以上をインストール

```
$ brew install gcc
```


## References

- ![Improving Distributional Similarity with Lessons Learned from Word Embeddings](https://transacl.org/ojs/index.php/tacl/article/view/570)
