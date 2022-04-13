## StratifiedKFold
- 目的数が不均衡データの場合、データの均衡を保ってfoldを作成する
- 回帰問題に関してはビン化した後の目的変数に適用する
- ビン化はスタージェスの公式などを利用する
```
# StratifiedKFold
# model_selection module of scikit-learn
import pandas as pd
from sklearn import model_selection

# train.csv --> 7セル目の学習データ（1000サンプル）
df = df_train
df["stratified_kfold"] = -1

# データをランダムにシャッフルする
df = df.sample(frac=1).reset_index(drop=True)

# y = df.target.values
y = df.quality.values

# StratifiedKFold classの初期化 (k=5)
kf = model_selection.StratifiedKFold(n_splits=5)

for fold, (trn_, val_) in enumerate(kf.split(X=df, y=y)):
    df.loc[val_, 'stratified_kfold'] = fold

df.to_csv("/content/drive/MyDrive/study/Kaggle_grandmaster/2_cross_validation/train_stratified_kfold", index=False)
```
```
import numpy as np
import pandas as pd
from sklearn import datasets
from sklearn import model_selection

def create_folds(data):
    data["kfold"] = -1
    
    data = data.sample(frac=1).reset_index(drop=True)
    
    # Styrge's Ruleによって，ビンの数を計算する
    num_bins = np.floor(1 + np.log2(len(data))).astype(int)
    
    data.loc[:, "bins"] = pd.cut(data["target"], bins=num_bins, labels=False)
    
    kf = model_selection.StratifiedKFold(n_splits=5)
    
    for f, (t_, v_) in enumerate(kf.split(X=data, y=data.bins.values)):
        data.loc[v_, "kfold"] = f
        
    # bins columnをdropする
    # data = data.drop("bins", axis=1)
    
    return data


# サンプル数=15000，特徴量の数=100，ターゲット=1
X, y = datasets.make_regression(n_samples=15000, n_features=100, n_targets=1)

df = pd.DataFrame(X, columns=[f"f_{i}" for i in range(X.shape[1])])
df.loc[:, "target"] = y

df = create_folds(df)
引用：kagglegrandmasterに学ぶ機械学習実践アプローチ第2章
```
