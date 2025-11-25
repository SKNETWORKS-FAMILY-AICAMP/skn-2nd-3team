"""
n_trials에 따른 앙상블 모델 성능 비교 스크립트

ensemble_strategy = ['voting', 'stacking'] 두 가지 앙상블 모델에 대해
n_trials를 30부터 450까지 30씩 증가시키면서 튜닝한 모델의 성능을 비교합니다.
"""

import pandas as pd
import numpy as np
import os
from typing import Dict, List

try:
    from tqdm import tqdm
    HAS_TQDM = True
except ImportError:
    HAS_TQDM = False
    def tqdm(iterable, desc=None):
        return iterable

from src.preprocessing import load_data, preprocess_pipeline, feature_engineering_pipeline, drop_column
from src.cv import stratified_kfold_split
from src.ensemble import train_stacking_ensemble, train_voting_ensemble, evaluate_model

# 환경 변수 IBSQR_USE_GPU=0 로 설정하면 강제로 CPU 사용
# LightGBM은 항상 CPU 모드로 실행되므로 기본값을 False로 설정
USE_GPU = os.environ.get("IBSQR_USE_GPU", "0").lower() in {"1", "true", "yes"}


def run_single_experiment(
    df: pd.DataFrame,
    ensemble_strategy: str,
    n_trials: int,
    cv_strategy: str = 'stratified_kfold',
    target_col: str = "Attrition_Binary",
    use_gpu: bool = False
) -> Dict[str, float]:
    """
    단일 실험 실행: 주어진 ensemble_strategy와 n_trials로 모델 학습 및 평가
    
    Args:
        df: 전처리된 데이터프레임
        ensemble_strategy: 'voting' 또는 'stacking'
        n_trials: Optuna 튜닝 시도 횟수
        cv_strategy: CV 전략
        target_col: 타겟 컬럼명
        use_gpu: XGBoost/LightGBM 학습 시 GPU 사용 여부
    
    Returns:
        Dict[str, float]: 평가 지표 딕셔너리
    """
    
    # 피처와 타겟 분리
    features = df.drop(columns=[target_col]).columns.tolist()
    X_full = df[features]
    y_full = df[target_col]
    
    # 전체 데이터로 튜닝하여 최적 파라미터 찾기
    if ensemble_strategy == 'stacking':
        tuned_model, tuned_params = train_stacking_ensemble(
            X_full, y_full,
            cv_strategy=cv_strategy,
            tuning_strategy='optuna',
            n_trials=n_trials,
            return_params=True,
            use_gpu=use_gpu
        )
    elif ensemble_strategy == 'voting':
        tuned_model, tuned_params = train_voting_ensemble(
            X_full, y_full,
            cv_strategy=cv_strategy,
            tuning_strategy='optuna',
            n_trials=n_trials,
            return_params=True,
            use_gpu=use_gpu
        )
    else:
        raise ValueError(f"Unknown ensemble_strategy: {ensemble_strategy}")
    
    # CV 설정
    if cv_strategy == 'stratified_kfold':
        folds = stratified_kfold_split(df, target_col=target_col, n_splits=5, shuffle=True, random_state=42)
    else:
        from sklearn.model_selection import train_test_split
        train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[target_col], random_state=42)
        folds = [(train_df.index.tolist(), test_df.index.tolist())]
    
    # 각 폴드에서 평가 (튜닝된 파라미터 사용)
    cv_results = []
    
    for train_idx, val_idx in folds:
        # 데이터 분할
        X_train = df.loc[train_idx, features]
        y_train = df.loc[train_idx, target_col]
        X_val = df.loc[val_idx, features]
        y_val = df.loc[val_idx, target_col]
        
        # 모델 학습 (튜닝된 파라미터 사용)
        if ensemble_strategy == 'stacking':
            model = train_stacking_ensemble(
                X_train, y_train,
                cv_strategy=cv_strategy,
                tuning_strategy=None,
                best_params=tuned_params,
                use_gpu=use_gpu
            )
        elif ensemble_strategy == 'voting':
            model = train_voting_ensemble(
                X_train, y_train,
                cv_strategy=cv_strategy,
                tuning_strategy=None,
                best_params=tuned_params,
                use_gpu=use_gpu
            )
        
        # 평가 (출력 없이)
        metrics = evaluate_model(
            model, X_val, y_val,
            print_report=False
        )
        
        cv_results.append(metrics)
    
    # CV 결과 평균 계산
    summary = {}
    for metric in cv_results[0].keys():
        values = [r[metric] for r in cv_results]
        summary[metric] = np.mean(values)
    
    return summary


def main():
    """메인 실행 함수"""
    
    print(f"\n{'='*80}")
    print("🔬 n_trials에 따른 앙상블 모델 성능 비교 실험")
    print(f"{'='*80}\n")
    
    # 데이터 로드 및 전처리
    print("1️⃣ 데이터 로드 및 전처리...")
    df = load_data()
    df = preprocess_pipeline(df)
    df = feature_engineering_pipeline(df)
    print(f"   ✅ 전처리 완료: {df.shape}\n")
    
    # 실험 설정
    ensemble_strategies = ['voting', 'stacking']
    n_trials_list = list(range(30, 451, 30))  # 30, 60, 90, ..., 450
    
    print(f"2️⃣ 실험 설정:")
    print(f"   - 앙상블 전략: {ensemble_strategies}")
    print(f"   - n_trials: {n_trials_list}")
    print(f"   - 연산 장치: {'GPU' if USE_GPU else 'CPU'}")
    print(f"   - 총 실험 수: {len(ensemble_strategies) * len(n_trials_list)}개\n")
    
    # 결과 저장 경로 설정
    save_dir = 'results/parameter_tuning'
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'n_trials_comparison.csv')
    
    # 기존 결과 파일이 있으면 로드
    if os.path.exists(save_path):
        print(f"📂 기존 결과 파일 발견: {save_path}")
        existing_results = pd.read_csv(save_path, encoding='utf-8-sig')
        print(f"   - 기존 실험 수: {len(existing_results)}개")
        print(f"   - 마지막 실험: {existing_results.iloc[-1]['ensemble_strategy']} - n_trials={existing_results.iloc[-1]['n_trials']}\n")
    else:
        print(f"📂 새로운 실험 시작\n")
        existing_results = pd.DataFrame()
    
    # 각 조합에 대해 실험 실행
    total_experiments = len(ensemble_strategies) * len(n_trials_list)
    experiment_num = 0
    
    for ensemble_strategy in ensemble_strategies:
        print(f"\n{'='*80}")
        print(f"📊 앙상블 전략: {ensemble_strategy.upper()}")
        print(f"{'='*80}\n")
        
        iterator = tqdm(n_trials_list, desc=f"{ensemble_strategy} 진행 중") if HAS_TQDM else n_trials_list
        for n_trials in iterator:
            experiment_num += 1
            
            # 이미 실험이 완료되었는지 확인
            if not existing_results.empty:
                already_done = existing_results[
                    (existing_results['ensemble_strategy'] == ensemble_strategy) & 
                    (existing_results['n_trials'] == n_trials)
                ]
                if not already_done.empty:
                    print(f"\n[{experiment_num}/{total_experiments}] {ensemble_strategy} - n_trials={n_trials}")
                    print(f"   ⏭️  이미 완료된 실험 - 건너뜀")
                    continue
            
            print(f"\n[{experiment_num}/{total_experiments}] {ensemble_strategy} - n_trials={n_trials}")
            
            try:
                # 실험 실행
                metrics = run_single_experiment(
                    df=df,
                    ensemble_strategy=ensemble_strategy,
                    n_trials=n_trials,
                    cv_strategy='stratified_kfold',
                    use_gpu=USE_GPU
                )
                
                # 결과 저장
                result_row = {
                    'ensemble_strategy': ensemble_strategy,
                    'n_trials': n_trials,
                    'accuracy': metrics['accuracy'],
                    'roc_auc': metrics['roc_auc'],
                    'pr_auc': metrics['pr_auc'],
                    'f1': metrics['f1'],
                    'recall': metrics['recall'],
                    'precision': metrics['precision'],
                    'device': 'gpu' if USE_GPU else 'cpu'
                }
                
                print(f"   ✅ 완료 - Accuracy: {metrics['accuracy']:.4f}, F1: {metrics['f1']:.4f}, ROC-AUC: {metrics['roc_auc']:.4f}")
                
            except Exception as e:
                print(f"   ❌ 오류 발생: {str(e)}")
                # 오류 발생 시에도 결과에 추가 (NaN 값으로)
                result_row = {
                    'ensemble_strategy': ensemble_strategy,
                    'n_trials': n_trials,
                    'accuracy': np.nan,
                    'roc_auc': np.nan,
                    'pr_auc': np.nan,
                    'f1': np.nan,
                    'recall': np.nan,
                    'precision': np.nan,
                    'device': 'gpu' if USE_GPU else 'cpu'
                }
            
            # 기존 결과에 새 결과 추가하여 즉시 저장
            new_result_df = pd.DataFrame([result_row])
            if existing_results.empty:
                existing_results = new_result_df
            else:
                existing_results = pd.concat([existing_results, new_result_df], ignore_index=True)
            
            # CSV 파일에 저장
            existing_results.to_csv(save_path, index=False, encoding='utf-8-sig')
            print(f"   💾 결과 저장 완료: {save_path}")
    
    print(f"\n{'='*80}")
    print("✅ 모든 실험 완료!")
    print(f"{'='*80}")
    print(f"\n📁 최종 결과 저장 위치: {save_path}")
    print(f"\n📊 결과 요약:")
    print(existing_results.groupby('ensemble_strategy').agg({
        'accuracy': ['mean', 'std'],
        'f1': ['mean', 'std'],
        'roc_auc': ['mean', 'std']
    }))
    print(f"\n")


if __name__ == '__main__':
    main()

