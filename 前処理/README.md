期間を指定する前処理
#19年9月23日〜29日のデータの抽出
A_period = purchase_article[(purchase_article['datetime'] > dt.datetime(2019,9,22)) 
& (purchase_article['datetime'] < dt.datetime(2019,9,30))]
