## 評価指標の最適化
- 19年9月23日〜29日のデータの抽出
```
A_period = purchase_article[(purchase_article['datetime'] > dt.datetime(2019,9,22)) 
& (purchase_article['datetime'] < dt.datetime(2019,9,30))]
```
