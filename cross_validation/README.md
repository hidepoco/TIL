## StratifiedKFold
- 19年9月23日〜29日のデータの抽出
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
