import pandas as pd
from sklearn.preprocessing import OrdinalEncoder, StandardScaler
from sklearn.compose import ColumnTransformer

UNSW_ATTACK_MAP = {
    'Normal':         'Normal',
    'DoS':            'DoS',
    'Reconnaissance': 'Probe',
    'Generic':        'R2L',
    'Exploits':       'R2L',
    'Fuzzers':        'R2L',
    'Analysis':       'R2L',
    'Backdoor':       'U2R',
    'Shellcode':      'U2R',
    'Worms':          'U2R',
}

# Columns to drop — not useful for classification
DROP_COLS = ['id', 'label']

# Categorical columns in UNSW-NB15
CATEGORICAL_COLS = ['proto', 'service', 'state']


def load_and_preprocess_unsw(
    train_path='UNSW_NB15_training-set.csv',
    test_path='UNSW_NB15_testing-set.csv'
):
    train_df = pd.read_csv(train_path)
    test_df  = pd.read_csv(test_path)

    print(f"UNSW raw train: {train_df.shape} | test: {test_df.shape}")

    # Map attack categories
    train_df['attack_category'] = train_df['attack_cat'].map(UNSW_ATTACK_MAP).fillna('Unknown')
    test_df['attack_category']  = test_df['attack_cat'].map(UNSW_ATTACK_MAP).fillna('Unknown')

    # Drop columns we don't need
    train_df = train_df.drop(columns=DROP_COLS + ['attack_cat'])
    test_df  = test_df.drop(columns=DROP_COLS + ['attack_cat'])

    print("Train class distribution:")
    print(train_df['attack_category'].value_counts())
    print("\nTest class distribution:")
    print(test_df['attack_category'].value_counts())

    # Separate features and labels
    X_train_raw = train_df.drop('attack_category', axis=1)
    y_train     = train_df['attack_category']
    X_test_raw  = test_df.drop('attack_category', axis=1)
    y_test      = test_df['attack_category']

    # Numeric columns = everything that isn't categorical
    numeric_cols = [c for c in X_train_raw.columns if c not in CATEGORICAL_COLS]

    # Build preprocessor — fit on train only
    preprocessor = ColumnTransformer(transformers=[
        ('cat', OrdinalEncoder(
            handle_unknown='use_encoded_value',
            unknown_value=-1
        ), CATEGORICAL_COLS),
        ('num', StandardScaler(), numeric_cols)
    ])
    preprocessor.fit(X_train_raw)

    # Apply to both splits
    feature_names = CATEGORICAL_COLS + numeric_cols

    X_train = pd.DataFrame(
        preprocessor.transform(X_train_raw),
        columns=feature_names
    )
    X_test = pd.DataFrame(
        preprocessor.transform(X_test_raw),
        columns=feature_names
    )

    print(f"\nProcessed — Train: {X_train.shape} | Test: {X_test.shape}")
    print(f"Classes: {sorted(y_train.unique())}")

    return X_train, X_test, y_train, y_test, preprocessor