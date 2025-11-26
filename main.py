"""
간결하고 명확한 머신러닝 파이프라인
"""

import pandas as pd
import numpy as np
import os
import joblib
from typing import Dict, List, Optional

from src.cv import stratified_kfold_split, kfold_split
from src.ensemble import train_stacking_ensemble, train_voting_ensemble, train_logistic_regression, evaluate_model
from src.preprocessing import load_data, preprocess_pipeline, feature_engineering_pipeline, drop_column


def run(
    df: pd.DataFrame,
    target_col: str = "Attrition_Binary",
    is_preprocess: bool = True,
    is_feature_engineering: bool = True,
    cv_strategy: str = 'stratified_kfold',  # 'stratified_kfold', 'kfold', None
    tuning_strategy: str = None,  # None, 'optuna', 'grid_search', 'random_search'
    ensemble_strategy: str = 'stacking',  # 'stacking', 'voting', 'logistic'
    is_save: bool = True,
) -> Dict:
    """
    머신러닝 파이프라인 실행
    
    💡 핵심 개선:
    - 튜닝은 한 번만 수행 (전체 데이터로)
    - CV 평가는 튜닝된 파라미터로 수행
    """
    
    print(f"\n{'='*80}")
    print("🚀 머신러닝 파이프라인 시작")
    print(f"{'='*80}\n")
    
    # 1️⃣ 전처리
    print("1️⃣ 데이터 전처리...")
    if is_preprocess:
        df = preprocess_pipeline(df)
    else:
        df = drop_column(df)
    
    if is_feature_engineering:
        print("2Feature Engineering...")
        df = feature_engineering_pipeline(df)
    
    print(f"   ✅ 전처리 완료: {df.shape}")
    
    # 피처와 타겟 분리
    features = df.drop(columns=[target_col]).columns.tolist()
    X_full = df[features]
    y_full = df[target_col]
    
    # 2️⃣ 튜닝 (한 번만!)
    tuned_params = None  # 튜닝된 파라미터 저장용
    if tuning_strategy is not None:
        print(f"\n2️⃣ 하이퍼파라미터 튜닝 ({tuning_strategy})")
        print("   ⚡ 전체 데이터로 한 번만 튜닝...")
        
        # 전체 데이터로 튜닝하여 최적 파라미터 찾기
        if ensemble_strategy == 'stacking':
            tuned_model, tuned_params = train_stacking_ensemble(
                X_full, y_full,
                cv_strategy=cv_strategy,
                tuning_strategy=tuning_strategy,
                n_trials=240,  # 필요시 조정
                return_params=True
            )
        elif ensemble_strategy == 'voting':
            tuned_model, tuned_params = train_voting_ensemble(
                X_full, y_full,
                cv_strategy=cv_strategy,
                tuning_strategy=tuning_strategy,
                n_trials=120,
                return_params=True
            )
        
        print("   ✅ 튜닝 완료! 최적 파라미터 찾음")
    else:
        print(f"\n2️⃣ 튜닝 스킵 (기본 파라미터 사용)")
    
    # 3️⃣ CV 설정
    print(f"\n3️⃣ CV 전략: {cv_strategy or '단순 분할'}")
    if cv_strategy == 'stratified_kfold':
        folds = stratified_kfold_split(df, target_col=target_col, n_splits=5, shuffle=True, random_state=42)
    elif cv_strategy == 'kfold':
        folds = kfold_split(df, n_splits=5, shuffle=True, random_state=42)
    else:
        # CV 없이 단순 분할
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[target_col], random_state=42)
        folds = [(train_df.index.tolist(), test_df.index.tolist())]
    
    # 4️⃣ 각 폴드에서 평가 (튜닝 안 함!)
    print(f"\n4️⃣ 모델 평가 (각 폴드)")
    print(f"   모델: {ensemble_strategy}")
    print(f"   {'✅ 튜닝된 파라미터 사용' if tuning_strategy else '기본 파라미터 사용'}\n")
    
    cv_results = []
    models = []
    
    for fold_num, (train_idx, val_idx) in enumerate(folds, 1):
        print(f"\n{'─'*60}")
        print(f"📍 Fold {fold_num}/{len(folds)}")
        print(f"{'─'*60}")
        
        # 데이터 분할
        X_train = df.loc[train_idx, features]
        y_train = df.loc[train_idx, target_col]
        X_val = df.loc[val_idx, features]
        y_val = df.loc[val_idx, target_col]
        
        # 모델 학습 (튜닝 없이!)
        if ensemble_strategy == 'stacking':
            model = train_stacking_ensemble(
                X_train, y_train,
                cv_strategy=cv_strategy,
                tuning_strategy=None,  # 👈 튜닝 안 함!
                best_params=tuned_params  # 👈 튜닝된 파라미터 재사용
            )
        elif ensemble_strategy == 'voting':
            model = train_voting_ensemble(
                X_train, y_train,
                cv_strategy=cv_strategy,
                tuning_strategy=None,  # 👈 튜닝 안 함!
                best_params=tuned_params  # 👈 튜닝된 파라미터 재사용
            )
        else:  # logistic
            model = train_logistic_regression(X_train, y_train)
        
        # 평가
        metrics = evaluate_model(
            model, X_val, y_val,
            fold_num=fold_num,
            n_splits=len(folds)
        )
        
        cv_results.append(metrics)
        models.append(model)
    
    # 5️⃣ 결과 요약
    print(f"\n{'='*80}")
    print("📊 최종 결과 요약")
    print(f"{'='*80}")
    
    summary = {}
    for metric in cv_results[0].keys():
        values = [r[metric] for r in cv_results]
        summary[metric] = {
            'mean': np.mean(values),
            'std': np.std(values)
        }
        if summary[metric]['mean'] > 0:  # ROC-AUC가 0이면 스킵
            print(f"{metric:12s}: {summary[metric]['mean']:.4f} ± {summary[metric]['std']:.4f}")
    print(f"{'='*80}\n")
    
    if is_save:
        # 6️⃣ 최종 모델 학습 (전체 데이터)
        print("6️⃣ 최종 모델 학습 (전체 데이터)...")
        
        if ensemble_strategy == 'stacking':
            final_model = train_stacking_ensemble(
                X_full, y_full,
                cv_strategy=cv_strategy,
                tuning_strategy=None,  # 👈 튜닝 안 함
                best_params=tuned_params  # 👈 튜닝된 파라미터 재사용
            )
        elif ensemble_strategy == 'voting':
            final_model = train_voting_ensemble(
                X_full, y_full,
                cv_strategy=cv_strategy,
                tuning_strategy=None,  # 👈 튜닝 안 함
                best_params=tuned_params  # 👈 튜닝된 파라미터 재사용
            )
        else:
            final_model = train_logistic_regression(X_full, y_full)
    
    # 7️⃣ 모델 저장
    if is_save:
        print(f"\n7️⃣ 모델 저장...")
        save_dir = 'results/Final_Model'
        os.makedirs(save_dir, exist_ok=True)
        model_path = os.path.join(save_dir, f'Exclude_Features_selection_00_{ensemble_strategy}_model.joblib')
        joblib.dump(final_model, model_path)
        print(f"   ✅ 저장 완료: {model_path}")
    
    print(f"\n{'='*80}")
    print("✅ 파이프라인 완료!")
    print(f"{'='*80}\n")
    
    return {
        'cv_results': cv_results,
        'summary': summary,
        'final_model': final_model,
        'best_fold_model': models[np.argmax([r['f1'] for r in cv_results])]
    }


if __name__ == '__main__':
    
    # 데이터 로드
    df = load_data()
    
    # 실행
    results = run(
        df=df,
        is_preprocess=True,
        is_feature_engineering=False,
        cv_strategy='stratified_kfold',  # 'stratified_kfold', 'kfold', None
        tuning_strategy='optuna',  # None, 'optuna', 'grid_search', 'random_search'
        ensemble_strategy='stacking',  # 'stacking', 'voting', 'logistic'
        is_save=True
    )