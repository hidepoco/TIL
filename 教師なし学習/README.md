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
![image](https://user-images.githubusercontent.com/73774284/172050279-1abc54ef-1211-4b42-a5c2-6ecc5e7253f6.png)

## スクリープロットで次元削減数を探索する

```
# ワインデータの取得
df_wine=pd.read_csv("https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data", header=None)
df_wine.columns = ["class", "Alcohol", "Malic acid", "Ash", "Alcalinity of ash","Magnesium", "Total phenols", "Flavanoids", "Nonflavanoid phenols","Proanthocyanins", "Color intensity", "Hue","OD280/OD315 of diluted wines", "Proline"]
display(df_wine.shape)
display(df_wine.head())
```
```
# ワインデータのPCA結果
from sklearn.decomposition import PCA
from sklearn import preprocessing
import numpy as np
sc=preprocessing.StandardScaler()
X = df_wine.iloc[:, 1:]
X_norm=sc.fit_transform(X)
 
pca = PCA(random_state=0)
X_pc = pca.fit_transform(X_norm)
df_pca = pd.DataFrame(X_pc, columns=["PC{}".format(i + 1) for i in range(len(X_pc[0]))])
print("主成分の数: ", pca.n_components_)
print("保たれている情報: ", round(np.sum(pca.explained_variance_ratio_),2))
display(df_pca.head())
```
![image](https://user-images.githubusercontent.com/73774284/172050346-bbf19d81-7ebc-480d-87b0-32d77538521b.png)

```
# 固有値の確認
# 標準化している際、固有値が固有値が1以上のものを使うというのがシンプルな方法
pd.DataFrame(np.round(pca.explained_variance_, 2), index=["PC{}".format(x + 1) for x in range(len(df_pca.columns))], columns=["固有値"])
```
```
# 固有値のスクリープロット
# 先ほど確認した固有値1のラインをラインを引く
# 安定する（今回のケースでは今回のケースでは8）までを利用することが望ましい
line = np.ones(14)
plt.plot(np.append(np.nan, pca.explained_variance_), "s-")
plt.plot(line, "s-")
plt.xlabel("PC")
plt.ylabel("explained_variance")
plt.xticks( np.arange(1, 14, 1))
plt.grid()
plt.show()
```
![image](https://user-images.githubusercontent.com/73774284/172050406-3b9bb390-9875-4c4c-b5a4-829ea4a73d65.png)

## 寄与率で次元削減数を探索
```
# 寄与率
pd.DataFrame(np.round(pca.explained_variance_ratio_,2), index=["PC{}".format(x + 1) for x in range(len(df_pca.columns))], columns=["寄与率"])
```
![image](https://user-images.githubusercontent.com/73774284/172050486-d52b863d-e7bb-4954-abd5-8ab40fadbd19.png)

```
# 累積寄与率の可視化
import matplotlib.ticker as ticker
# 基準線の作成
line = np.full(14, 0.9)
plt.gca().get_xaxis().set_major_locator(ticker.MaxNLocator(integer=True))
plt.plot([0] + list( np.cumsum(pca.explained_variance_ratio_)), "-o")
plt.xlabel("PC")
plt.ylabel("cumulative contribution rate")
plt.yticks( np.arange(0, 1.1, 0.1))
plt.plot(line, "s-") 
plt.grid()
plt.show()
```
![image](https://user-images.githubusercontent.com/73774284/172050534-b832603b-2df1-43e2-b015-12f3627a9eef.png)

```
# 累積寄与率を指定してPCAを実行
sc=preprocessing.StandardScaler()
X = df_wine.iloc[:, 1:]
X_norm=sc.fit_transform(X)
# 累積寄与率が寄与率が90%を超えるPC8までを結果として表示
pca = PCA(n_components=0.9, random_state=0)
X_pc = pca.fit_transform(X_norm)
df_pca = pd.DataFrame(X_pc, columns=["PC{}".format(i + 1) for i in range(len(X_pc[0]))])
print("主成分の数: ", pca.n_components_) 
print("保たれている情報: ", round(np.sum(pca.explained_variance_ratio_),2))
display(df_pca.head())
```
![image](https://user-images.githubusercontent.com/73774284/172050587-3aa9cad9-a527-4e90-85b7-3a64aa005a4e.png)

## UMAPによる次元削減
- 処理速度が早く四次元以上の圧縮にも対応している次元削減のトレンド手法の一つ
- 非線形次元削減にも使用できる
```
!pip3 install umap-learn

# UMAPとt-SNEを実行
import umap
 
start_time_tsne = time.time()
X_reduced = TSNE(n_components=2, random_state=0).fit_transform(digits.data)
interval_tsne = time.time() - start_time_tsne
 
start_time_umap = time.time()
embedding = umap.UMAP(n_components=2, random_state=0).fit_transform(digits.data)
interval_umap = time.time() - start_time_umap
 
print("tsne : {}s".format(np.round(interval_tsne,2)))
print("umap : {}s".format(np.round(interval_umap,2)))
```

```
# UMAPとt-SNEの結果
plt.figure(figsize=(10,8))
plt.subplot(2, 1, 1)
for each_label in digits.target_names:
    c_plot_bool = digits.target == each_label
    plt.scatter(X_reduced[c_plot_bool, 0], X_reduced[c_plot_bool, 1], label="{}".format(each_label))
plt.legend(loc="upper right")
plt.xlabel("tsne-1")
plt.ylabel("tsne-2")
 
plt.subplot(2, 1, 2)
for each_label in digits.target_names:
    c_plot_bool = digits.target == each_label
    plt.scatter(embedding[c_plot_bool, 0], embedding[c_plot_bool, 1], label="{}".format(each_label))
plt.legend(loc="upper right")
plt.xlabel("umap-1")
plt.ylabel("umap-2")
plt.show()
```
![image](https://user-images.githubusercontent.com/73774284/172050829-fa9538af-929c-4626-a4d6-d670015417b6.png)

## Umapで最適なn_neighborsを探索
- 大きくするとマクロな、小さくするとミクロな構造を結果に反映反映
- 2~100の間の値を選択が推奨。デフォルトはデフォルトは15で設定されている
```
# 最適な最適なn_neighborsを探索する関数（二次元表示）
# n_neighborsを2,5,30,50,100ごとにUMAPを実施して結果を二次元に可視化する関数
def create_2d_umap(target_X, y, y_labels, n_neighbors_list= [2, 15, 30, 50, 100]):
    fig, axes = plt.subplots(nrows=1, ncols=len(n_neighbors_list),figsize=(5*len(n_neighbors_list), 4))
    for i, (ax, n_neighbors) in enumerate(zip(axes.flatten(), n_neighbors_list)):
        start_time = time.time()
        # 二次元から変更する際はする際は2の値を変更変更
        mapper = umap.UMAP(n_components=2, random_state=0, n_neighbors=n_neighbors)
        Y = mapper.fit_transform(target_X)
        for each_label in y_labels:
            c_plot_bool = y == each_label
            ax.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], label="{}".format(each_label))
        end_time = time.time()
        ax.legend(loc="upper right")
        ax.set_title("n_neighbors: {}".format(n_neighbors))
        print("n_neighbors {} is {:.2f} seconds.".format(n_neighbors, end_time - start_time))
    plt.show()
```
```
# UMAPの結果
create_2d_umap(digits.data, digits.target, digits.target_names)
```
![image](https://user-images.githubusercontent.com/73774284/172050927-f88213ad-49a5-4e55-b0e9-46ccf90331ab.png)

```
# 最適な最適なn_neighborsを探索する関数（3次元表示）
def create_3d_umap(target_X, y, y_labels, n_neighbors_list= [2, 15, 30, 50, 100]):
    fig = plt.figure(figsize=(5*len(n_neighbors_list),4))
    for i, n_neighbors in enumerate(n_neighbors_list):
        ax = fig.add_subplot(1, len(n_neighbors_list), i+1, projection="3d")
        start_time = time.time()
        mapper = umap.UMAP(n_components=3, random_state=0, n_neighbors=n_neighbors)
        Y = mapper.fit_transform(target_X)
        for each_label in y_labels:
            c_plot_bool = y == each_label
            ax.scatter(Y[c_plot_bool, 0], Y[c_plot_bool, 1], label="{}".format(each_label))
        end_time = time.time()
        ax.legend(loc="upper right")
        ax.set_title("n_neighbors_list: {}".format(n_neighbors))
        print("n_neighbors_list {} is {:.2f} seconds.".format(n_neighbors, end_time - start_time))
    plt.show()
```
    
```
create_3d_umap(digits.data, digits.target, digits.target_names)
```
   
![image](https://user-images.githubusercontent.com/73774284/172050968-3c1ecd56-eebf-432e-9646-dce6bee7c04b.png)

## PCAとUMAPを組み合わせて次元削減を実施実施
- 高次元データを扱う場合PCA結果をさらにUMAPで次元削減することで良い結果になるケースがある
```
# PCAの結果
# 累積寄与率が累積寄与率が99%になるなるPC41までの結果が表示される
pca = PCA(n_components=0.99, random_state=0)
X_pc = pca.fit_transform(digits.data)
df_pca = pd.DataFrame(X_pc, columns=["PC{}".format(i + 1) for i in range(len(X_pc[0]))])
print("主成分の数: ", pca.n_components_) 
print("保たれている情報: ", np.sum(pca.explained_variance_ratio_))
display(df_pca.head())
```
![image](https://user-images.githubusercontent.com/73774284/172051107-f66deb66-baa7-4030-82e5-367d54703db1.png)

```
# 左下のn_neighbors5の結果が良い
create_2d_umap(digits.data, digits.target, digits.target_names, [5,10,15])
create_2d_umap(df_pca, digits.target, digits.target_names, [5,10,15])
```    
![image](https://user-images.githubusercontent.com/73774284/172051165-81cc3d0f-d884-443a-af89-7b21a2ef77ff.png)

    
    
    
