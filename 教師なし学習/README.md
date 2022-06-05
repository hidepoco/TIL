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

## VBGMM（変分混合ガウスモデル）
- クラスタ数がわからない場合に有効
- ベイズ推定に基づき確率分布を推定しながらクラスタ数や分布の形状を決定
```
plt.figure(figsize=(10,3))
plt.scatter(x,y, c=z_km.labels_)
plt.scatter(z_km.cluster_centers_[:,0],z_km.cluster_centers_[:,1],s=250, marker="*",c="red")
plt.suptitle("k-means")
plt.show
 
from sklearn import mixture
vbgmm = mixture.BayesianGaussianMixture(n_components=10, random_state=0)
vbgmm=vbgmm.fit(X_norm)
labels=vbgmm.predict(X_norm)
 
plt.figure(figsize=(10,3))
plt.scatter(x,y, c=labels)
plt.suptitle("vbgmm")
plt.show
```
## VBGMMで最適なクラスタ数を探索
```
#クラスタごと各データの分布の確認確認
# n=10で指定していたが指定していたが3で事足りそうであることがわかる
x_tick =np.array([1,2,3,4,5,6,7,8,9,10])
plt.figure(figsize=(10,2))
plt.bar(x_tick, vbgmm.weights_, width=0.7, tick_label=x_tick)
plt.suptitle("vbgmm_weights")
plt.show

# VBGMMでクラスタリング（クラスタ数数3）
vbgmm = mixture.BayesianGaussianMixture(n_components=3, random_state=0)
vbgmm = vbgmm.fit(X_norm)
labels = vbgmm.predict(X_norm)

plt.figure(figsize=(10, 3))
plt.scatter(x,y,c=labels)
plt.suptitle("vbgmm")
plt.show
```
![image](https://user-images.githubusercontent.com/73774284/171982096-cf122a7d-dcf2-46f3-b52f-10a71b1ddf0a.png)

```
x_tick = np.array([1,2,3])
plt.figure(figsize=(10,2))
plt.bar(x_tick, vbgmm.weights_, width=0.7, tick_label=x_tick)
plt.suptitle("vbgmm_weights")
plt.show
```
![image](https://user-images.githubusercontent.com/73774284/171982046-2b45b7a9-3d12-4cdc-9e70-6e948285224a.png)

## 次元削減
- データの可視化
- データ容量の節約
- 特徴量の作成

## PCA
```
データのダウンロード
import pandas as pd
from sklearn.datasets import load_iris
iris = load_iris()
df=pd.DataFrame(iris.data, columns=iris.feature_names)
df["target"] = iris.target
df.loc[df["target"]==0, "target_name"] = "setosa"
df.loc[df["target"]==1, "target_name"] = "versicolor"
df.loc[df["target"]==2, "target_name"] = "virginica"
df.head()
```
```
# PCA結果
from sklearn.decomposition import PCA
import numpy as np
pca = PCA(random_state=0)
X_pc = pca.fit_transform(df.iloc[:, 0:4])
df_pca = pd.DataFrame(X_pc, columns=["PC{}".format(i + 1) for i in range(len(X_pc[0]))])
print("主成分の数: ", pca.n_components_) 
print("保たれている情報: ", np.sum(pca.explained_variance_ratio_))
display(df_pca.head())
```
![image](https://user-images.githubusercontent.com/73774284/172050141-1514a2c1-9a2d-43e5-8599-5a0daf6cf56c.png)

```# PC1ととPC2を可視化
sns.scatterplot(x="PC1", y="PC2", data=df_pca, hue=df["target_name"])
```
![image](https://user-images.githubusercontent.com/73774284/172050170-9dd21f94-c6d3-46f0-a366-89d3c75e5473.png)

## 主成分に関する用語
![image](https://user-images.githubusercontent.com/73774284/172050205-1eaf9abf-4807-43a4-af8d-3fb5019affc4.png)

```
# 各主成分と元データの相関図
# https://pythondatascience.plavox.info/seaborn/heatmap（ヒートマップの引数について）
# https://qiita.com/kenichiro_nishioka/items/8e307e164a4e0a279734（figについての説明）
import seaborn as sns
fig = plt.figure(figsize=(8,6))
ax = fig.add_subplot(111)
sns.heatmap(pca.components_,
           cmap="Blues",
           annot=True,
           annot_kws={"size": 14},
           fmt=".2f",
           xticklabels=["SepalLength", "SepalWidth", "PetalLength", "PetalLength"],
           yticklabels=["PC1", "PC2", "PC3", "PC4"],
           ax=ax)
plt.show()
```
