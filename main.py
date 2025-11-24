"""
Churn 예측 모델 선택 파이프라인
- 모든 기본 모델 학습 및 비교
- 앙상블 모델 학습 및 비교
- 최적 모델 자동 선택
"""
from src.preprocessing import load_data, preprocess_data
from src.model_selection import compare_all_models, compare_ensemble_models, select_best_model
from sklearn.model_selection import train_test_split


def main():
    """
    모델 선택 파이프라인 실행
    """
    print("=" * 80)
    print("Churn 예측 모델 선택 파이프라인 시작")
    print("=" * 80)
    
    # 1. 데이터 로드 및 전처리
    print("\n[1단계] 데이터 로드 및 전처리 중...")
    df = load_data()
    df = preprocess_data(df)
    print(f"데이터 shape: {df.shape}")
    
    # 2. 데이터 분리 (Train/Val/Test)
    print("\n[2단계] 데이터 분리 중...")
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
    print("\n[3단계] 기본 모델 성능 비교 중...")
    base_results_df, base_results = compare_all_models(
        X_train_final, y_train_final, X_test, y_test, cv=5
    )
    
    # 4. 앙상블 모델들 비교
    print("\n[4단계] 앙상블 모델 성능 비교 중...")
    ensemble_results = compare_ensemble_models(
        X_train_final, y_train_final, X_val, y_val, X_test, y_test
    )
    
    # 5. 최적 모델 선택
    print("\n[5단계] 최적 모델 선택 중...")
    best_model = select_best_model(
        base_results, ensemble_results, metric='combined_auc'
    )
    
    print("\n" + "=" * 80)
    print("모델 선택 파이프라인 완료!")
    print("=" * 80)
    
    return best_model, base_results_df, ensemble_results


if __name__ == '__main__':
    best_model, base_results, ensemble_results = main()