from src.cv import split_train_test, kfold_split, stratified_kfold_split
from src.ensemble import train_logistic_regression, evaluate_model, train_stacking_ensemble, train_voting_ensemble
from src.preprocessing import load_data, preprocess_pipeline, drop_column, replace_nan_value, add_feature_engineering, select_features

# 데이터 불러오기 및 전처리

def run(
    self,
    df: pd.DataFrame,
    target_col: str = "Attrition_Binary",
    is_preprocess: bool = True,
    is_feature_engineering: bool = True,
    is_cv: bool = True,
    is_tuning: bool = True,
    is_ensemble: bool = True,
    is_save: bool = True,
    ) -> Dict:
    """
    Run complete pipeline

    Args:
        df (_type_): _description_
        is_preprocess (bool, optional): _description_. Defaults to True.
        is_feature_engineering (bool, optional): _description_. Defaults to True.
        is_ensemble (bool, optional): _description_. Defaults to True.
        is_tuning (bool, optional): _description_. Defaults to True.
        is_save (bool, optional): _description_. Defaults to True.

    Returns:
        Dict: _description_
    """

    if is_preprocess:
        df = preprocess_pipeline(df)
    else:
        df = drop_column(df)

    if is_feature_engineering:
        df = feature_engineering_pipeline(df)
    
    if is_cv:
        folds = stratified_kfold_split(df, target_col=target_col, n_splits=5, shuffle=True, random_state=42)
        for i, (train_idx, test_idx) in enumerate(folds):
            # fold별 train 데이터 분할
            X_train = df.loc[train_idx, features]
            y_train = df.loc[train_idx, target_col]

            # fold별 test 데이터 분할
            X_test = df.loc[test_idx, features]
            y_test = df.loc[test_idx, target_col]

            if is_ensemble:
                model = train_stacking_ensemble(X_train, y_train) # ensemble.py
            else:
                model = train_logistic_regression(X_train, y_train)
            evaluate_model(model, X_test, y_test)
    else:
        # 일반적인 Fold
        X_train, X_test, y_train, y_test = split_train_test(df, target_col)
        if is_ensemble:
            model = train_stacking_ensemble(X_train, y_train) # ensemble.py
        else:
            model = train_logistic_regression(X_train, y_train)
        evaluate_model(model, X_test, y_test)

    if is_save:
        save_dir = 'results/Final_Model'
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, 'final_model.joblib')
        joblib.dump(model, model_path)
        print(f"모델이 저장되었습니다: {model_path}")   

    else:
        print("모델 저장을 하지 않았습니다.")
if __name__ == '__main__':
    df = load_data()
    run(
        df,
        is_preprocess=True,
        is_feature_engineering=True,
        is_cv=True,
        is_tuning=True,
        is_ensemble=True,
        is_save=True,
        )

    # 모델 저장
    import joblib
    import os
    
    save_dir = 'results/Final_Model'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'final_model.joblib')
    joblib.dump(model, model_path)
    print(f"모델이 저장되었습니다: {model_path}")