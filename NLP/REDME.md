## 類似度の高い単語10件取得
- indexは1から表示させる
```
simularities = model.most_similar("United_States")
pd.DataFrame(
    simularities,
    columns = ['単語', '類似度'],
    index = np.arange(len(simularities)) + 1
)
```
## 加法構成性によるアナロジー
- Spain”の単語ベクトルから”Madrid”のベクトルを引き，”Athens”のベクトルを足したベクトルを計算し，そのベクトルと類似度の高い10語とその類似度を出力せよ．
```
simularities = model.most_similar(positive=['Spain', 'Athens'], negative=['Madrid'])
pd.DataFrame(
    simularities,
    columns = ['単語', '類似度'],
    index = np.arange(len(simularities)) + 1
)
```
## n-gram
- 文章の単語単位の分割
- nが増えてもn2と同様のパターンで増やしていく
```
s = "新型コロナのワクチンが早く欲しい"
def unigram(s):
  return [(s[i],) for i in range(len(s))]
```

```
# 2-gram
def bigram(s):
  return [(s[i],s[i+1]) for i in range(len(s)) if i+1 <len(s)]
```


# n-gram関数
```
def ngram(s,n):
  """ngramを計算して返却
    Args:
        s(str): 解析対象文字列
        n(int):n
    Returns:
        ngram(list):ngramリスト
    """
  ngram = []
  for i in range(len(s)):
    if i + n >=len(s):
      break
    tpl = tuple()
    for j in range(n):
      tpl += (s[i+j],)
    ngram.append(tpl)
  return ngram

s = "新型コロナウイルスのワクチンが早く欲しい。"
print(ngram(s, 3))
```


# 文字列とリストを渡すとリスト内の要素を文字列から削除する関数
```
def exclude(s,list):
  for exc in list:
    s = s.replace(exc,"")
  return s
exclude_list = ["失敗"]
exclude("私の辞書に「失敗」という文字はない、「成功」に向かって一つ実験結果を得ただけだ。",exclude_list)
```
# 正解ラベルの作成
- カテゴリ名とリストの検索マップをcat2idとして字書型で格納
- 辞書内包表記
- https://atmarkit.itmedia.co.jp/ait/articles/2107/13/news019.html
```
categories = list(set(df['カテゴリ']))
cat2id = {cat:i, cat in enumerate(categories)}
print(cat2id.items())
```
