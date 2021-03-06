# -*- coding: utf-8 -*-

import sys
import os
import nagisa
from gensim.models import word2vec
import numpy as np
from statistics import mean, median,variance,stdev

def main(argv=sys.argv):
    '''
    argv[1] モデルファイルが入っているパス
    argv[2] 会議のテキストのパス
    '''
    if len(argv) != 3:
        print('Usage: {} [input text]'.format(argv[0]))
        sys.exit(1)

    # モデル取得
    model = word2vec.Word2Vec.load(argv[1])

    i = 0
    for x in os.listdir(argv[2]):
        # 会議のテキスト取得
        f = open(argv[2] + x)
        text = f.read()  # ファイル終端まで全て読んだデータを返す
        f.close()
        analyze(text, model, i)
        i += 5

def analyze(text, model, i):
    node = wakachi(text)

    # 自責
    print(wakachi('僕のせい、結局は 自分のせい 悪かった 自分が'))
    self_condemnation = get_vector(node, wakachi('僕のせい、結局は 自分のせい 悪かった 自分が'), model)

    # さらけ出し
    print(wakachi('知らない 失敗した 恥ずかしい 辛い うまくいかない ミスった できないんですよね できない 失敗しました'))
    sarakedashi = get_vector(node, wakachi('知らない 失敗した 恥ずかしい 辛い うまくいかない ミスった できないんですよね できない 失敗しました'), model)

    # 成長を信じる
    print(wakachi('変わった 変わりそう 成長 大丈夫 挑戦を後押しした 可能性がある できる'))
    growth_mind = get_vector(node, wakachi('変わった 変わりそう 成長 大丈夫 挑戦を後押しした 可能性がある できる'), model)

    # アンラーン
    print(wakachi('今思うと やってみたい 試してみる 試す 教えて欲しい'))
    unlearn = get_vector(node, wakachi('今思うと やってみたい 試してみる 試す 教えて欲しい'), model)

    print([i, self_condemnation, sarakedashi, growth_mind, unlearn])


def wakachi(text):
    return nagisa.extract(text, extract_postags=['名詞']).words

# テキストのベクトルを計算
def get_vector(words1, words2, model):
    sum_vec = []
    word_count = 0

    for word1 in words1:
        for word2 in words2:
            try:
                sum_vec.append(cos_sim(model.wv[word1], model.wv[word2]))
            except KeyError as instance:
                pass
            word_count += 1
    arr = np.sort(sum_vec)[::-1][:100]
    return np.average(arr)


# cos類似度を計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == '__main__':
    main()
