import os, urllib.request, pandas as pd, ssl
ssl._create_default_https_context = ssl._create_unverified_context
DATA_DIR = "data"
TRAIN_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTrain%2B.txt"
TEST_URL = "https://raw.githubusercontent.com/defcom17/NSL_KDD/master/KDDTest%2B.txt"
COLUMNS = ['duration', 'protocol_type', 'service', 'flag', 'src_bytes', 'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot', 'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell', 'su_attempted', 'num_root', 'num_file_creations', 'num_shells', 'num_access_files', 'num_outbound_cmds', 'is_host_login', 'is_guest_login', 'count', 'srv_count', 'serror_rate', 'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate', 'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate', 'dst_host_srv_diff_host_rate', 'dst_host_serror_rate', 'dst_host_srv_serror_rate', 'dst_host_rerror_rate', 'dst_host_srv_rerror_rate', 'label', 'difficulty_level']

if __name__ == "__main__":
    os.makedirs(DATA_DIR, exist_ok=True)
    for url, fname in [(TRAIN_URL, "KDDTrain+.txt"), (TEST_URL, "KDDTest+.txt")]:
        if not os.path.exists(os.path.join(DATA_DIR, fname)):
            print(f"Downloading {fname}...")
            urllib.request.urlretrieve(url, os.path.join(DATA_DIR, fname))
    for fname in ["KDDTrain+.txt", "KDDTest+.txt"]:
        df = pd.read_csv(os.path.join(DATA_DIR, fname), header=None, names=COLUMNS)
        if 'difficulty_level' in df.columns: df = df.drop(columns=['difficulty_level'])
        df.to_csv(os.path.join(DATA_DIR, fname.replace('.txt', '.csv')), index=False)
        print(f"Saved {fname.replace('.txt', '.csv')}: {df.shape}")
    print("Done!")
