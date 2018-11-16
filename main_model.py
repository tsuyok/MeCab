# -*- coding: utf-8 -*-

import sys
import nagisa
from gensim.models import word2vec
import numpy as np
import subprocess

import MeCab


def main(argv=sys.argv):
    '''
    argv[1] コーパス作成元のテキストファイルが入ったディレクトリ
    '''

    if len(argv) != 2:
        print('Usage: {} [input text]'.format(argv[0]))
        sys.exit(1)

    # テキストからコーパス作成
    subprocess.check_call("find " + argv[1] + " | grep wiki | awk '{system(\"cat \"$0\" >> wiki.txt\")}'")

    # MeCab
    subprocess.check_call("mecab -d /usr/lib/x86_64-linux-gnu/mecab/dic/mecab-ipadic-neologd wiki.txt -o wiki_wakati.txt")

    # 文字コード
    subprocess.check_call("nkf -w --overwrite wiki_wakati.txt")

    # Word2Vec モデル化
    sentences = word2vec.Text8Corpus('./wiki_wakati.txt')

    model = word2vec.Word2Vec(sentences, size=200, min_count=20, window=15)
    model.save("./wiki.model")


if __name__ == '__main__':
    main()