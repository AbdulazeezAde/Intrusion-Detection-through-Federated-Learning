import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List
import torch
from torch.utils.data import DataLoader, TensorDataset
import warnings
warnings.filterwarnings('ignore')

# Define exact column names for NSL-KDD (41 features + label + difficulty)
COLUMNS = [
    'duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land',
    'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised',
    'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
    'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count',
    'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate',
    'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
    'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate',
    'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
    'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate',
    'label', 'difficulty_level'
]

def load_and_preprocess_nslkdd(train_path: str, test_path: str) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Load and preprocess NSL-KDD dataset with robust type handling."""
    print(f"Loading data from {train_path} and {test_path}...")
    
    # Load with low_memory=False to handle mixed types
    train_df = pd.read_csv(train_path, names=COLUMNS, low_memory=False)
    test_df = pd.read_csv(test_path, names=COLUMNS, low_memory=False)

    # Drop difficulty_level
    if 'difficulty_level' in train_df.columns:
        train_df = train_df.drop(columns=['difficulty_level'])
    if 'difficulty_level' in test_df.columns:
        test_df = test_df.drop(columns=['difficulty_level'])

    # Separate features and labels
    X_train = train_df.drop(columns=['label']).copy()
    y_train = (train_df['label'] != 'normal').astype(int).values
    X_test = test_df.drop(columns=['label']).copy()
    y_test = (test_df['label'] != 'normal').astype(int).values

    # Identify columns
    categorical_columns = ['protocol_type', 'service', 'flag']
    numerical_columns = [c for c in X_train.columns if c not in categorical_columns]

    # FIX: Force categorical to string, numerical to float
    for col in categorical_columns:
        X_train[col] = X_train[col].astype(str)
        X_test[col] = X_test[col].astype(str)
    
    for col in numerical_columns:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce').fillna(0)
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce').fillna(0)

    # One-Hot Encoding
    encoder = OneHotEncoder(sparse_output=False, drop='first', handle_unknown='ignore')
    X_train_cat = encoder.fit_transform(X_train[categorical_columns])
    X_test_cat = encoder.transform(X_test[categorical_columns])

    # Scale Numerical
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_columns].values.astype(float))
    X_test_num = scaler.transform(X_test[numerical_columns].values.astype(float))

    # Combine
    X_train_final = np.hstack([X_train_num, X_train_cat]).astype(np.float32)
    X_test_final = np.hstack([X_test_num, X_test_cat]).astype(np.float32)

    print(f"Preprocessing complete. Train: {X_train_final.shape}, Test: {X_test_final.shape}")
    return X_train_final, X_test_final, y_train, y_test

def partition_data_for_fl(X: np.ndarray, y: np.ndarray, num_clients: int, iid: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition data for FL simulation."""
    partitions = []
    if iid:
        indices = np.random.permutation(len(X))
        X, y = X[indices], y[indices]
        chunk = len(X) // num_clients
        for i in range(num_clients):
            start, end = i * chunk, (i + 1) * chunk if i < num_clients - 1 else len(X)
            partitions.append((X[start:end], y[start:end]))
    else:
        # Non-IID simple shard
        for i in range(num_clients):
            idx = np.random.choice(len(X), len(X)//num_clients, replace=False)
            partitions.append((X[idx], y[idx]))
    return partitions

def create_dataloader(X: np.ndarray, y: np.ndarray, batch_size: int = 64, shuffle: bool = True) -> DataLoader:
    """Create PyTorch DataLoader."""
    return DataLoader(
        TensorDataset(torch.FloatTensor(X), torch.LongTensor(y)),
        batch_size=batch_size, shuffle=shuffle
    )
