## 代表的なアルゴリズム一覧とその用途
![image](https://user-images.githubusercontent.com/73774284/171978527-970ead70-72c2-463c-a641-fc5de02cdf88.png)

## 変数の散布図
- indexは1から表示させる
```
import seaborn as sns
df_temp = df_iris.copy()
sns.pairplot(df_temp)
```
![image](https://user-images.githubusercontent.com/73774284/171981278-a73f2384-9ab6-401f-b218-27e284015029.png)

```
# k-meansの実行
from sklearn.cluster import KMeans
model = KMeans(n_clusters=3, random_state=0, init="random")
cls_data = df_iris.copy()
model.fit(cls_data)

# クラスタの予測結果取得
cluster = model.predict(cls_data)
print(cluster)

# クラスタの予測結果取得（図）
cls_data["cluster"] = cluster
sns.pairplot(cls_data, hue="cluster")
```
![image](https://user-images.githubusercontent.com/73774284/171981349-884e1d66-7d54-4b2d-aa2e-e14a3f0411fe.png)

## クラスタリング結果の評価
```
# 最良の場合いずれも1を返す
from sklearn.metrics import accuracy_score
from sklearn.metrics import adjusted_rand_score
ari = "ARI: {:.2f}".format(adjusted_rand_score(iris.target, cls_data["cluster"]))
accuracy = "Accuracy: {:.2f}".format(accuracy_score(iris.target, cls_data["cluster"]))
print(ari)
print(accuracy)
```
