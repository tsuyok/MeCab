# -*- coding: utf-8 -*-

import sys
import nagisa
from gensim.models import word2vec
import numpy as np


def main(argv=sys.argv):
    # Word2Vec モデル化
    sentences = word2vec.Text8Corpus(argv[1])

    model = word2vec.Word2Vec(sentences, size=200, min_count=20, window=15)
    model.save("./wiki.model")


if __name__ == '__main__':
    main()