# TIL
日々の学びについて記載

# 使い回しコード集
　#データの結合
　purchase_article = pd.merge(transactions,articles,how="left",on="article_id")
 
　#日付データの特徴量抽出
 import datetime as dt
#日付をobject型からdatetime型へ変換し年月日の特徴量を抽出
purchase_article['datetime'] = pd.to_datetime(purchase_article['t_dat'], format='%Y-%m-%d')
# 年
purchase_article["year"] = purchase_article["datetime"].dt.year
# 月
purchase_article["month"] = purchase_article["datetime"].dt.month
# 日
purchase_article["day"] = purchase_article["datetime"].dt.day
# 年月
purchase_article["year_month"] = purchase_article["datetime"].dt.strftime("%Y%m")
 　
