# -*- coding: utf-8 -*-

import sys
import nagisa
from gensim.models import word2vec
import numpy as np


def main(argv=sys.argv):
    if len(argv) != 2:
        print('Usage: {} [input text]'.format(argv[0]))
        sys.exit(1)

    # コーチングのテキストを取得
    f = open(argv[1])
    text = f.read()  # ファイル終端まで全て読んだデータを返す
    f.close()

    node = wakachi(text)

    # モデル取得
    model = word2vec.Word2Vec.load("./wiki.model")

    # ベクトルを計算

    # コーチング
    coaching = get_vector(node, model)

    # 自責
    print(wakachi('僕のせい、結局は 自分のせい 悪かった 自分が'))
    self_condemnation = get_vector(wakachi('僕のせい、結局は 自分のせい 悪かった 自分が'), model)

    print(cos_sim(coaching, self_condemnation))

    # さらけ出し
    print(wakachi('知らない 失敗した 恥ずかしい 辛い うまくいかない ミスった できないんですよね できない 失敗しました'))
    sarakedashi = get_vector(wakachi('知らない 失敗した 恥ずかしい 辛い うまくいかない ミスった できないんですよね できない 失敗しました'), model)

    print(cos_sim(coaching, sarakedashi))

    # 成長を信じる
    print(wakachi('変わった 変わりそう 成長 大丈夫 挑戦を後押しした 可能性がある できる'))
    growth_mind = get_vector(wakachi('変わった 変わりそう 成長 大丈夫 挑戦を後押しした 可能性がある できる'), model)

    print(cos_sim(coaching, growth_mind))

    # アンラーン
    print(wakachi('今思うと やってみたい 試してみる 試す 教えて欲しい'))
    unlearn = get_vector(wakachi('今思うと やってみたい 試してみる 試す 教えて欲しい'), model)

    print(cos_sim(coaching, unlearn))


def wakachi(text):
    return nagisa.extract(text, extract_postags=['名詞','動詞']).words

# テキストのベクトルを計算
def get_vector(words, model):
    sum_vec = np.zeros(200)
    word_count = 0

    for word in words:
        try:
            sum_vec += model.wv[word]
        except KeyError as instance:
            pass
        word_count += 1

    return sum_vec / word_count


# cos類似度を計算
def cos_sim(v1, v2):
    return np.dot(v1, v2) / (np.linalg.norm(v1) * np.linalg.norm(v2))

if __name__ == '__main__':
    main()