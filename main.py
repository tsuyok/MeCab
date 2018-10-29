# -*- coding: utf-8 -*-

import MeCab
import sys

mecab = MeCab.Tagger ('-d /usr/local/lib/mecab/dic/mecab-ipadic-neologd')

if __name__ == '__main__':
    if len(sys.argv) != 3:
        print('Usage: {} [input video] [output text]'.format(sys.argv[0]))
        sys.exit(1)

    f = open(sys.argv[1])
    text = f.read()  # ファイル終端まで全て読んだデータを返す
    f.close()

    mecab.parse('')#文字列がGCされるのを防ぐ
    node = mecab.parseToNode(text)

    results = []
    while node:
        #単語を取得
        word = node.surface
        #品詞を取得
        pos = node.feature.split(",")[1]
        results.append('{0} , {1}'.format(word, pos))
        #次の単語に進める
        node = node.next

    with open(sys.argv[2], 'w') as output:
        output.write('\n'.join(results))
