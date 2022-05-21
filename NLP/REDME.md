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
