import pandas as pd
from sklearn.preprocessing import LabelEncoder

def load_and_preprocess_data(train_path="KDDTrain+.txt", test_path="KDDTest+.txt"):
    """
    Loads NSL-KDD dataset from txt files, adds column names, handles categorical encoding,
    and groups attack types into 5 broad categories.

    Args:
        train_path (str): Relative path to training data.
        test_path (str): Relative path to test data.

    Returns:
        X_train (pd.DataFrame): Preprocessed features for training.
        X_test (pd.DataFrame): Preprocessed features for testing.
        y_train (pd.Series): Target labels for training.
        y_test (pd.Series): Target labels for testing.
    """
    columns = [
        'duration', 'protocol_type', 'service', 'flag', 'src_bytes',
        'dst_bytes', 'land', 'wrong_fragment', 'urgent', 'hot',
        'num_failed_logins', 'logged_in', 'num_compromised', 'root_shell',
        'su_attempted', 'num_root', 'num_file_creations', 'num_shells',
        'num_access_files', 'num_outbound_cmds', 'is_host_login',
        'is_guest_login', 'count', 'srv_count', 'serror_rate',
        'srv_serror_rate', 'rerror_rate', 'srv_rerror_rate', 'same_srv_rate',
        'diff_srv_rate', 'srv_diff_host_rate', 'dst_host_count',
        'dst_host_srv_count', 'dst_host_same_srv_rate',
        'dst_host_diff_srv_rate', 'dst_host_same_src_port_rate',
        'dst_host_srv_diff_host_rate', 'dst_host_serror_rate',
        'dst_host_srv_serror_rate', 'dst_host_rerror_rate',
        'dst_host_srv_rerror_rate', 'label', 'difficulty'
    ]

    attack_map = {
        'normal': 'Normal',
        'neptune': 'DoS', 'back': 'DoS', 'land': 'DoS', 'pod': 'DoS',
        'smurf': 'DoS', 'teardrop': 'DoS', 'mailbomb': 'DoS',
        'apache2': 'DoS', 'processtable': 'DoS', 'udpstorm': 'DoS',
        'ipsweep': 'Probe', 'nmap': 'Probe', 'portsweep': 'Probe',
        'satan': 'Probe', 'mscan': 'Probe', 'saint': 'Probe',
        'ftp_write': 'R2L', 'guess_passwd': 'R2L', 'imap': 'R2L',
        'multihop': 'R2L', 'phf': 'R2L', 'spy': 'R2L', 'warezclient': 'R2L',
        'warezmaster': 'R2L', 'sendmail': 'R2L', 'named': 'R2L',
        'snmpgetattack': 'R2L', 'snmpguess': 'R2L', 'xlock': 'R2L',
        'xsnoop': 'R2L', 'worm': 'R2L',
        'buffer_overflow': 'U2R', 'loadmodule': 'U2R', 'perl': 'U2R',
        'rootkit': 'U2R', 'httptunnel': 'U2R', 'ps': 'U2R',
        'sqlattack': 'U2R', 'xterm': 'U2R'
    }

    # Load data without headers and explicitly specify columns
    train_df = pd.read_csv(train_path, names=columns)
    test_df = pd.read_csv(test_path, names=columns)

    # Drop 'difficulty' column
    train_df = train_df.drop('difficulty', axis=1)
    test_df = test_df.drop('difficulty', axis=1)

    # Map target variable (label -> attack_category)
    train_df['attack_category'] = train_df['label'].map(attack_map)
    test_df['attack_category'] = test_df['label'].map(attack_map)

    # Some newer/rarer attacks might not be in our explicit map, let's treat any unmapped ones properly or map to their known class if they occur
    # Usually NSL-KDD test set contains some novel attacks. By assignment specs we map using the provided dict.
    # To avoid NaNs from novel attacks in test set, let's keep them as 'Unknown' or just drop them. The prompt says "map labels using attack_map above".
    # I'll default to fillna('Unknown') just in case, removing NaNs.
    train_df['attack_category'] = train_df['attack_category'].fillna('Unknown')
    test_df['attack_category'] = test_df['attack_category'].fillna('Unknown')

    # Drop old 'label' column
    train_df = train_df.drop('label', axis=1)
    test_df = test_df.drop('label', axis=1)

    # Encode categorical columns
    categorical_cols = ['protocol_type', 'service', 'flag']
    
    # We combine train and test to fit LabelEncoder on all possible values and avoid unseen label errors
    for col in categorical_cols:
        le = LabelEncoder()
        # fitting on combination of both to handle all unique cases
        le.fit(pd.concat([train_df[col], test_df[col]]))
        train_df[col] = le.transform(train_df[col])
        test_df[col] = le.transform(test_df[col])

    # Separate features and targets
    X_train = train_df.drop('attack_category', axis=1)
    y_train = train_df['attack_category']
    
    X_test = test_df.drop('attack_category', axis=1)
    y_test = test_df['attack_category']

    return X_train, X_test, y_train, y_test

if __name__ == "__main__":
    X_tr, X_te, y_tr, y_te = load_and_preprocess_data()
    print("Preprocessing completed successfully.")
    print(f"Train Shape: {X_tr.shape}, Test Shape: {X_te.shape}")
