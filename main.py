from src.cv import split_train_test, kfold_split, stratified_kfold_split
from src.ensemble import train_logistic_regression, evaluate_model
from src.preprocessing import load_data, preprocess_data   

# 데이터 불러오기 및 전처리

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df) # preprocessing.py

    features = df.columns.drop('Attrition_Binary')
    target_col = 'Attrition_Binary'

# ========================= Train Test Split =========================
    # X_train, X_test, y_train, y_test = split_train_test( # cv.py
    #     df, target_col,
    #     test_size=0.2, 
    #     random_state=42
    #     )

# ========================= KFold =========================
    # folds = kfold_split(df, n_splits=5, shuffle=True, random_state=42)

# ========================= Stratify CV Fold =========================    
    folds = stratified_kfold_split(df, target_col=target_col, n_splits=5, shuffle=True, random_state=42)

    for i, (train_idx, test_idx) in enumerate(folds):
        # fold별 train 데이터 분할
        X_train = df.loc[train_idx, features]
        y_train = df.loc[train_idx, target_col]

        # fold별 test 데이터 분할
        X_test = df.loc[test_idx, features]
        y_test = df.loc[test_idx, target_col]

        # fold별 모델 학습 및 평가
        model = train_logistic_regression(X_train, y_train) # ensemble.py
        evaluate_model(model, X_test, y_test)

    # 모델 저장
    import joblib
    import os
    
    save_dir = 'results/Final_Model'
    os.makedirs(save_dir, exist_ok=True)
    model_path = os.path.join(save_dir, 'final_model.joblib')
    joblib.dump(model, model_path)
    print(f"모델이 저장되었습니다: {model_path}")