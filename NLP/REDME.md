simularities = model.most_similar("United_States")
pd.DataFrame(
    simularities,
    columns = ['単語', '類似度'],
    index = np.arange(len(simularities)) + 1
)


## 期間を指定する前処理
- 19年9月23日〜29日のデータの抽出
```
A_period = purchase_article[(purchase_article['datetime'] > dt.datetime(2019,9,22)) 
& (purchase_article['datetime'] < dt.datetime(2019,9,30))]
```
