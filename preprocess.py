import pandas as pd
import numpy as np
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

COLUMNS = [
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

ATTACK_MAP = {
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

CATEGORICAL_COLS = ['protocol_type', 'service', 'flag']


def load_raw(path, columns):
    df = pd.read_csv(path, names=columns)
    df = df.drop('difficulty', axis=1)
    df['attack_category'] = df['label'].map(ATTACK_MAP).fillna('Unknown')
    df = df.drop('label', axis=1)
    return df


def build_preprocessor(train_df):
    """
    Fits a ColumnTransformer on training data only.
    Returns the fitted preprocessor.
    """
    numeric_cols = [c for c in train_df.columns
                    if c not in CATEGORICAL_COLS + ['attack_category']]

    # OrdinalEncoder: integers for tree models, handles unseen values gracefully
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ), CATEGORICAL_COLS),
        ('num', StandardScaler(), numeric_cols)
    ])

    X_train = train_df.drop('attack_category', axis=1)
    preprocessor.fit(X_train)
    return preprocessor, numeric_cols


def apply_preprocessor(preprocessor, df):
    X = df.drop('attack_category', axis=1)
    y = df['attack_category']

    X_transformed = preprocessor.transform(X)

    numeric_cols = [
        c for c in X.columns
        if c not in CATEGORICAL_COLS
    ]

    all_feature_names = CATEGORICAL_COLS + numeric_cols

    X_out = pd.DataFrame(
        X_transformed,
        columns=all_feature_names,
        index=df.index
    )

    return X_out, y

def load_and_preprocess_data(train_path="KDDTrain+.txt", test_path="KDDTest+.txt"):
    """
    Loads NSL-KDD, fits preprocessor on train only, applies to both splits.

    Returns:
        X_train, X_test, y_train, y_test, preprocessor
    """
    train_df = load_raw(train_path, COLUMNS)
    test_df = load_raw(test_path, COLUMNS)

    preprocessor, _ = build_preprocessor(train_df)

    X_train, y_train = apply_preprocessor(preprocessor, train_df)
    X_test, y_test = apply_preprocessor(preprocessor, test_df)

    print(f"Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Classes: {sorted(y_train.unique())}")
    return X_train, X_test, y_train, y_test, preprocessor


if __name__ == "__main__":
    load_and_preprocess_data()