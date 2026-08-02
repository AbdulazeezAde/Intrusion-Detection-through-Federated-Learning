from typing import Tuple, List
import pandas as pd, numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder

def load_and_preprocess_nslkdd(train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    train_df, test_df = pd.read_csv(train_path), pd.read_csv(test_path)
    for df in [train_df, test_df]:
        if 'difficulty_level' in df.columns: df.drop(columns=['difficulty_level'], inplace=True)
    y_train = (train_df['label'] != 'normal').astype(int).values
    y_test = (test_df['label'] != 'normal').astype(int).values
    X_train, X_test = train_df.drop(columns=['label']), test_df.drop(columns=['label'])
    cat_cols = ['protocol_type', 'service', 'flag']
    num_cols = [c for c in X_train.columns if c not in cat_cols]
    enc = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_cat_train, X_cat_test = enc.fit_transform(X_train[cat_cols]), enc.transform(X_test[cat_cols])
    scaler = StandardScaler()
    X_num_train = scaler.fit_transform(X_train[num_cols].astype(np.float32))
    X_num_test = scaler.transform(X_test[num_cols].astype(np.float32))
    return np.hstack([X_num_train, X_cat_train.astype(np.float32)]).astype(np.float32), \
           np.hstack([X_num_test, X_cat_test.astype(np.float32)]).astype(np.float32), y_train, y_test

def partition_data_for_fl(X, y, num_clients, iid=True):
    idx = np.random.permutation(len(X)) if iid else np.arange(len(X))
    X_s, y_s = X[idx], y[idx]
    splits = np.array_split(np.arange(len(X)), num_clients)
    return [(X_s[s], y_s[s]) for s in splits]
