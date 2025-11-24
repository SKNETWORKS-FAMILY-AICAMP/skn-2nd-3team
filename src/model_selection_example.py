"""
Model Selection 파이프라인 사용 예시
"""
from src.preprocessing import load_data, preprocess_data
from src.cv import split_train_test
from src.model_selection import compare_all_models, compare_ensemble_models, select_best_model
from sklearn.model_selection import train_test_split
import numpy as np


def main():
    # 1. 데이터 로드 및 전처리
    print("데이터 로드 중...")
    df = load_data()
    df = preprocess_data(df)
    
    # 2. 데이터 분리 (Train/Val/Test)
    X = df.drop('Attrition_Binary', axis=1)
    y = df['Attrition_Binary']
    
    # Train/Test 분리
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    # Train에서 Validation 분리
    X_train_final, X_val, y_train_final, y_val = train_test_split(
        X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
    )
    
    print(f"Train: {X_train_final.shape}, Val: {X_val.shape}, Test: {X_test.shape}")
    
    # 3. 기본 모델들 비교
    base_results_df, base_results = compare_all_models(
        X_train_final, y_train_final, X_test, y_test, cv=5
    )
    
    # 4. 앙상블 모델들 비교
    ensemble_results = compare_ensemble_models(
        X_train_final, y_train_final, X_val, y_val, X_test, y_test
    )
    
    # 5. 최적 모델 선택
    best_model = select_best_model(
        base_results, ensemble_results, metric='combined_auc'
    )
    
    print("\n최적 모델 선택 완료!")
    return best_model, base_results_df, ensemble_results


if __name__ == '__main__':
    best_model, base_results, ensemble_results = main()

