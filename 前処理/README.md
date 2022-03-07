## 期間を指定する前処理
- 19年9月23日〜29日のデータの抽出
```
A_period = purchase_article[(purchase_article['datetime'] > dt.datetime(2019,9,22)) 
& (purchase_article['datetime'] < dt.datetime(2019,9,30))]
```
## 日付をobject型からdatetime型へ変換し年月日の特徴量を抽出
```
purchase_article['datetime'] = pd.to_datetime(purchase_article['t_dat'], format='%Y-%m-%d')
# 年
purchase_article["year"] = purchase_article["datetime"].dt.year
# 月
purchase_article["month"] = purchase_article["datetime"].dt.month
# 日
purchase_article["day"] = purchase_article["datetime"].dt.day
# 年月
purchase_article["year_month"] = purchase_article["datetime"].dt.strftime("%Y%m")
```
                                                         
## カラムの要素をカウントし多い順に並び替える
- C_period期間中のarticle_idをカウントし、多い順に並び替える
```
C_order = C_period.groupby('article_id')['article_id'].count().sort_values(ascending=False)
```
## 検算
- article_idが866731001のデータの数のカウント
```
C_period[C_period['article_id'] == 866731001].count()
```
