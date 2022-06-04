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
