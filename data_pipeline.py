import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from typing import Tuple, List
import warnings
warnings.filterwarnings('ignore')

# Define the 41 features + label + difficulty (43 total columns in raw file)
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
    """
    Load and preprocess NSL-KDD dataset.
    Handles raw .txt files without headers.
    """
    print(f"Loading data from {train_path} and {test_path}...")
    
    # Load raw data
    try:
        train_df = pd.read_csv(train_path, header=None, names=COLUMNS)
        test_df = pd.read_csv(test_path, header=None, names=COLUMNS)
    except Exception as e:
        raise FileNotFoundError(f"Error loading dataset: {e}. Ensure paths are correct.")

    # Drop difficulty_level
    if 'difficulty_level' in train_df.columns:
        train_df.drop('difficulty_level', axis=1, inplace=True)
    if 'difficulty_level' in test_df.columns:
        test_df.drop('difficulty_level', axis=1, inplace=True)

    # Binary Label Mapping: normal=0, attack=1
    # Handle potential mixed types by converting to string first
    train_df['label'] = train_df['label'].astype(str).apply(lambda x: 0 if x == 'normal' else 1)
    test_df['label'] = test_df['label'].astype(str).apply(lambda x: 0 if x == 'normal' else 1)

    # Separate Features and Labels
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    X_train = train_df.drop('label', axis=1)
    X_test = test_df.drop('label', axis=1)

    # Identify Categorical and Numerical Columns
    categorical_cols = ['protocol_type', 'service', 'flag']
    numerical_cols = [c for c in X_train.columns if c not in categorical_cols]

    # Force numerical columns to numeric types (handling mixed type warnings)
    for col in numerical_cols:
        X_train[col] = pd.to_numeric(X_train[col], errors='coerce')
        X_test[col] = pd.to_numeric(X_test[col], errors='coerce')

    # Fill NaNs resulting from coercion
    X_train.fillna(0, inplace=True)
    X_test.fillna(0, inplace=True)

    # One-Hot Encoding for Categorical
    encoder = OneHotEncoder(sparse_output=False, handle_unknown='ignore')
    X_train_cat = encoder.fit_transform(X_train[categorical_cols])
    X_test_cat = encoder.transform(X_test[categorical_cols])

    # Standard Scaling for Numerical
    scaler = StandardScaler()
    X_train_num = scaler.fit_transform(X_train[numerical_cols])
    X_test_num = scaler.transform(X_test[numerical_cols])

    # Concatenate
    X_train_final = np.hstack([X_train_num, X_train_cat])
    X_test_final = np.hstack([X_test_num, X_test_cat])

    print(f"Preprocessing complete. Shape: Train={X_train_final.shape}, Test={X_test_final.shape}")
    return X_train_final, X_test_final, y_train, y_test

def partition_data_for_fl(X: np.ndarray, y: np.ndarray, num_clients: int, iid: bool = True) -> List[Tuple[np.ndarray, np.ndarray]]:
    """Partition data for FL simulation."""
    if iid:
        indices = np.random.permutation(len(X))
        X_shuffled, y_shuffled = X[indices], y[indices]
        split_size = len(X) // num_clients
        partitions = []
        for i in range(num_clients):
            start = i * split_size
            end = (i + 1) * split_size if i < num_clients - 1 else len(X)
            partitions.append((X_shuffled[start:end], y_shuffled[start:end]))
        return partitions
    else:
        # Simple Non-IID: Shard by label
        partitions = [[] for _ in range(num_clients)]
        unique_labels = np.unique(y)
        # Distribute labels round-robin
        for label in unique_labels:
            idx = np.where(y == label)[0]
            np.random.shuffle(idx)
            shards = np.array_split(idx, num_clients)
            for i, shard in enumerate(shards):
                partitions[i].append(shard)
        
        final_partitions = []
        for i in range(num_clients):
            all_idx = np.concatenate(partitions[i])
            final_partitions.append((X[all_idx], y[all_idx]))
        return final_partitions
