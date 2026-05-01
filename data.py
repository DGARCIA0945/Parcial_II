import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, LabelEncoder
from imblearn.combine import SMOTETomek
from torch.utils.data import Dataset, DataLoader
import torch

class UNSWNB15Dataset(Dataset):
    def __init__(self, X, y, seq_len=32):
        self.X, self.y = [], []
        for i in range(len(X) - seq_len):
            self.X.append(X[i:i+seq_len])
            self.y.append(y[i+seq_len-1])
        self.X = np.array(self.X, dtype=np.float32)
        self.y = np.array(self.y, dtype=np.int64)

    def __len__(self): return len(self.y)
    def __getitem__(self, idx):
        return torch.tensor(self.X[idx]), torch.tensor(self.y[idx])

def load_data(csv_paths, seq_len=32):
    dfs = [pd.read_csv(p, low_memory=False, encoding='utf-8-sig') for p in csv_paths]
    df  = pd.concat(dfs, ignore_index=True)

    label_col = 'label' if 'label' in df.columns else 'Label'
    y = df[label_col].values.astype(int)

    drop_cols = [label_col, 'attack_cat', 'id']
    df = df.drop(columns=[c for c in drop_cols if c in df.columns])

    for c in df.select_dtypes(include='object').columns:
        le = LabelEncoder()
        df[c] = le.fit_transform(df[c].astype(str))

    df = df.fillna(0)
    X  = df.values.astype(np.float32)

    n       = len(X)
    n_test  = int(n * 0.15)
    n_val   = int(n * 0.15)
    n_train = n - n_test - n_val

    X_train, y_train = X[:n_train], y[:n_train]
    X_val,   y_val   = X[n_train:n_train+n_val], y[n_train:n_train+n_val]
    X_test,  y_test  = X[n_train+n_val:], y[n_train+n_val:]

    scaler  = MinMaxScaler()
    X_train = scaler.fit_transform(X_train)
    X_val   = scaler.transform(X_val)
    X_test  = scaler.transform(X_test)

    sm = SMOTETomek(random_state=42)
    X_train, y_train = sm.fit_resample(X_train, y_train)

    train_ds = UNSWNB15Dataset(X_train, y_train, seq_len)
    val_ds   = UNSWNB15Dataset(X_val,   y_val,   seq_len)
    test_ds  = UNSWNB15Dataset(X_test,  y_test,  seq_len)

    return (DataLoader(train_ds, batch_size=32, shuffle=True),
            DataLoader(val_ds,   batch_size=64, shuffle=False),
            DataLoader(test_ds,  batch_size=64, shuffle=False),
            X_train.shape[1])
