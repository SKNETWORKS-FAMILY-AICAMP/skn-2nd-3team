from sklearn.model_selection import train_test_split, KFold, StratifiedKFold

def split_train_test(
    df,
    target_col='Attrition_Binary',
    test_size=0.2,
    random_state=42,
):
    """Split data into training and test sets using stratified sampling.

    Parameters
    ----------
    df : pandas.DataFrame
        The full dataset.
    target_col : str, default 'Attrition_Binary'
        Column name of the target variable.
    test_size : float, default 0.2
        Proportion of the dataset to include in the test split.
    random_state : int, default 42
        Random seed for reproducibility.

    Returns
    -------
    X_train, X_test, y_train, y_test : pandas.DataFrame / pandas.Series
        Split features and target.
    """
    X = df.drop(target_col, axis=1)
    y = df[target_col]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    return X_train, X_test, y_train, y_test


def kfold_split(df, n_splits=5, shuffle=False, random_state=None):
    """Generate K-Fold cross‑validation splits.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to split.
    n_splits : int, default 5
        Number of folds.
    shuffle : bool, default False
        Whether to shuffle before splitting.
    random_state : int or None, default None
        Random seed used when ``shuffle=True``.

    Returns
    -------
    list of tuple
        Each tuple contains ``(train_index, test_index)`` arrays.
    """
    kf = KFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(kf.split(df))


def stratified_kfold_split(df, target_col='Attrition_Binary', n_splits=5, shuffle=False, random_state=None):
    """Generate stratified K‑Fold splits preserving class distribution.

    Parameters
    ----------
    df : pandas.DataFrame
        The dataset to split.
    target_col : str, default 'Attrition_Binary'
        Column name of the target variable used for stratification.
    n_splits : int, default 5
        Number of folds.
    shuffle : bool, default False
        Whether to shuffle before splitting.
    random_state : int or None, default None
        Random seed used when ``shuffle=True``.

    Returns
    -------
    list of tuple
        Each tuple contains ``(train_index, test_index)`` arrays.
    """
    y = df[target_col]
    skf = StratifiedKFold(n_splits=n_splits, shuffle=shuffle, random_state=random_state)
    return list(skf.split(df, y))
