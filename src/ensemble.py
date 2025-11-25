"""
앙상블 모델 학습 및 평가 모듈

이 모듈은 다음 기능을 제공합니다:
- Voting Classifier (투표 기반 앙상블)
- Stacking Classifier (스태킹 앙상블)
- 모델 평가 함수
"""

from typing import Tuple, List, Optional, Dict, Any, Union
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, StackingClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, roc_auc_score, classification_report, 
    f1_score, recall_score, precision_score, average_precision_score
)
from sklearn.model_selection import StratifiedKFold, KFold
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier


####################################
# 🔧 CV 헬퍼 함수
####################################
def _get_cv_splitter(
    cv_method: Optional[str] = None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42
) -> Union[int, Any]:
    """
    CV 전략을 선택하여 적절한 CV splitter를 반환합니다.
    
    💡 cv.py의 함수들을 sklearn 호환 형식으로 변환
    
    Args:
        cv_method: CV 방법 선택
            - None or 'default': 기본 int 반환 (sklearn 자동 처리)
            - 'stratified_kfold': StratifiedKFold (불균형 데이터 추천!)
            - 'kfold': KFold
        n_splits: 폴드 수
        shuffle: 섞기 여부
        random_state: 랜덤 시드
    
    Returns:
        sklearn CV splitter 또는 int
        
    예시:
        >>> cv = _get_cv_splitter('stratified_kfold', n_splits=5)
        >>> # StratifiedKFold(n_splits=5, shuffle=True, random_state=42) 반환
    """
    
    if cv_method is None or cv_method == 'default':
        # sklearn이 자동으로 처리 (classification이면 StratifiedKFold 사용)
        return n_splits
    
    elif cv_method == 'stratified_kfold':
        # 클래스 비율 유지 (불균형 데이터에 적합)
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
    
    elif cv_method == 'kfold':
        # 일반 KFold
        return KFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
    
    else:
        raise ValueError(
            f"Unknown cv_method: {cv_method}\n"
            f"Available options: None, 'default', 'stratified_kfold', 'kfold'"
        )

####################################
# 📌 공통 함수: 기본 모델 생성 및 튜닝
####################################
def _create_base_models(
    scale_pos_weight: float,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    tuning_strategy: Optional[str] = None,
    cv: int = 5,
    n_trials: int = 50
) -> List[Tuple[str, Any]]:
    """
    앙상블에 사용할 기본 모델들(Random Forest, XGBoost, LightGBM)을 생성합니다.
    선택적으로 하이퍼파라미터 튜닝을 수행할 수 있습니다.
    
    💡 설계 철학:
    1. 기본 모드 (tuning_strategy=None): 빠른 프로토타이핑
    2. 튜닝 모드 (tuning_strategy='optuna' 등): 성능 최적화
    
    Args:
        scale_pos_weight (float): 클래스 불균형 처리 가중치
        X_train (Optional[pd.DataFrame]): 튜닝 시 필요한 훈련 데이터 (튜닝 안하면 None 가능)
        y_train (Optional[pd.Series]): 튜닝 시 필요한 타겟 데이터 (튜닝 안하면 None 가능)
        tuning_strategy (Optional[str]): 튜닝 방법
            - None: 기본 파라미터 사용 (빠름)
            - 'grid_search': 격자 탐색 (전수 조사, 느림)
            - 'random_search': 랜덤 탐색 (중간)
            - 'optuna': Optuna 베이지안 최적화 (추천!)
        cv (int): 교차 검증 폴드 수
        n_trials (int): Optuna/RandomSearch 시도 횟수
    
    Returns:
        List[Tuple[str, model]]: (모델이름, 모델객체) 튜플의 리스트
        
    예시:
        # 기본 모드 (빠름)
        >>> models = _create_base_models(scale_pos_weight=3.0)
        
        # 튜닝 모드 (느리지만 성능 좋음)
        >>> models = _create_base_models(
        ...     scale_pos_weight=3.0,
        ...     X_train=X_train,
        ...     y_train=y_train,
        ...     tuning_strategy='optuna',
        ...     n_trials=50
        ... )
    """
    
    # 🔍 튜닝 여부 확인
    if tuning_strategy is not None:
        if X_train is None or y_train is None:
            raise ValueError(
                "❌ 튜닝을 하려면 X_train과 y_train이 필요합니다!\n"
                "   _create_base_models(..., X_train=X, y_train=y, tuning_strategy='optuna')"
            )
        print(f"\n🔧 하이퍼파라미터 튜닝 모드: {tuning_strategy}")
        print(f"   데이터 크기: {X_train.shape}, CV={cv}, Trials={n_trials}")
        return _tune_base_models(
            X_train=X_train,
            y_train=y_train,
            scale_pos_weight=scale_pos_weight,
            tuning_strategy=tuning_strategy,
            cv=cv,
            n_trials=n_trials
        )
    
    # 📦 기본 모드: 미리 정의된 하이퍼파라미터 사용
    print("\n⚡ 기본 파라미터 모드 (튜닝 없음)")
    
    # 🌲 Random Forest 설정
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

    # 🚀 XGBoost 설정
    xgb_model = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        eval_metric="logloss",
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )

    # 💡 LightGBM 설정
    lgbm_model = LGBMClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
        n_jobs=-1,
        verbosity=-1
    )

    return [
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ]


####################################
# 🔧 하이퍼파라미터 튜닝 함수
####################################
def _tune_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    tuning_strategy: str = 'optuna',
    cv: int = 5,
    n_trials: int = 50
) -> List[Tuple[str, Any]]:
    """
    각 기본 모델의 하이퍼파라미터를 튜닝합니다.
    
    💡 튜닝 전략 비교:
    - grid_search: 모든 조합 시도 (완벽하지만 매우 느림)
    - random_search: 랜덤 샘플링 (빠르고 괜찮음)
    - optuna: 베이지안 최적화 (똑똑하고 효율적, 추천!)
    
    💡 각 모델별 중요 파라미터:
    - Random Forest: n_estimators, max_depth, min_samples_split
    - XGBoost: learning_rate, max_depth, subsample, colsample_bytree
    - LightGBM: learning_rate, num_leaves, max_depth
    """
    from src.tuner import grid_search_tuner, random_search_tuner, optuna_tuner
    import optuna
    
    tuned_models = []
    
    # 🌲 Random Forest 튜닝
    print("\n🌲 Random Forest 튜닝 중...")
    if tuning_strategy == 'optuna':
        def rf_factory(trial: optuna.Trial):
            return RandomForestClassifier(
                n_estimators=trial.suggest_int('n_estimators', 100, 500),
                max_depth=trial.suggest_int('max_depth', 5, 20),
                min_samples_split=trial.suggest_int('min_samples_split', 2, 10),
                class_weight='balanced',
                random_state=42,
                n_jobs=-1
            )
        rf_best, rf_params, rf_score = optuna_tuner(
            rf_factory, X_train, y_train, 
            cv=cv, n_trials=n_trials, scoring='recall'
        )
    elif tuning_strategy == 'grid_search':
        from src.tuner import grid_search_tuner
        param_grid = {
            'n_estimators': [100, 200, 300],
            'max_depth': [5, 10, 15],
            'min_samples_split': [2, 5, 10]
        }
        rf_best, rf_params, rf_score = grid_search_tuner(
            RandomForestClassifier(class_weight='balanced', random_state=42),
            param_grid, X_train, y_train, cv=cv, scoring='recall'
        )
    elif tuning_strategy == 'random_search':
        from src.tuner import random_search_tuner
        from scipy.stats import randint
        param_dist = {
            'n_estimators': randint(100, 500),
            'max_depth': randint(5, 20),
            'min_samples_split': randint(2, 10)
        }
        rf_best, rf_params, rf_score = random_search_tuner(
            RandomForestClassifier(class_weight='balanced', random_state=42),
            param_dist, X_train, y_train, 
            cv=cv, n_iter=n_trials, scoring='recall'
        )
    
    print(f"   ✅ 최적 파라미터: {rf_params}")
    print(f"   ✅ CV 점수: {rf_score:.4f}")
    tuned_models.append(('rf', rf_best))
    
    # 🚀 XGBoost 튜닝
    print("\n🚀 XGBoost 튜닝 중...")
    if tuning_strategy == 'optuna':
        def xgb_factory(trial: optuna.Trial):
            return XGBClassifier(
                n_estimators=trial.suggest_int('n_estimators', 100, 500),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                scale_pos_weight=scale_pos_weight,
                eval_metric="logloss",
                random_state=42,
                n_jobs=-1,
                verbosity=0
            )
        xgb_best, xgb_params, xgb_score = optuna_tuner(
            xgb_factory, X_train, y_train,
            cv=cv, n_trials=n_trials, scoring='recall'
        )
    elif tuning_strategy == 'grid_search':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'max_depth': [3, 5, 7]
        }
        xgb_best, xgb_params, xgb_score = grid_search_tuner(
            XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, verbosity=0),
            param_grid, X_train, y_train, cv=cv, scoring='recall'
        )
    elif tuning_strategy == 'random_search':
        from scipy.stats import uniform, randint
        param_dist = {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.29),
            'max_depth': randint(3, 10)
        }
        xgb_best, xgb_params, xgb_score = random_search_tuner(
            XGBClassifier(scale_pos_weight=scale_pos_weight, random_state=42, verbosity=0),
            param_dist, X_train, y_train,
            cv=cv, n_iter=n_trials, scoring='recall'
        )
    
    print(f"   ✅ 최적 파라미터: {xgb_params}")
    print(f"   ✅ CV 점수: {xgb_score:.4f}")
    tuned_models.append(('xgb', xgb_best))
    
    # 💡 LightGBM 튜닝
    print("\n💡 LightGBM 튜닝 중...")
    if tuning_strategy == 'optuna':
        def lgbm_factory(trial: optuna.Trial):
            return LGBMClassifier(
                n_estimators=trial.suggest_int('n_estimators', 100, 500),
                learning_rate=trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
                num_leaves=trial.suggest_int('num_leaves', 20, 100),
                max_depth=trial.suggest_int('max_depth', 3, 10),
                subsample=trial.suggest_float('subsample', 0.6, 1.0),
                colsample_bytree=trial.suggest_float('colsample_bytree', 0.6, 1.0),
                scale_pos_weight=scale_pos_weight,
                random_state=42,
                n_jobs=-1,
                verbosity=-1
            )
        lgbm_best, lgbm_params, lgbm_score = optuna_tuner(
            lgbm_factory, X_train, y_train,
            cv=cv, n_trials=n_trials, scoring='recall'
        )
    elif tuning_strategy == 'grid_search':
        param_grid = {
            'n_estimators': [100, 200, 300],
            'learning_rate': [0.01, 0.05, 0.1],
            'num_leaves': [31, 50, 70],
            'max_depth': [3, 5, 7]
        }
        lgbm_best, lgbm_params, lgbm_score = grid_search_tuner(
            LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42, verbosity=-1),
            param_grid, X_train, y_train, cv=cv, scoring='recall'
        )
    else:  # random_search
        from scipy.stats import uniform, randint
        param_dist = {
            'n_estimators': randint(100, 500),
            'learning_rate': uniform(0.01, 0.29),
            'num_leaves': randint(20, 100),
            'max_depth': randint(3, 10)
        }
        lgbm_best, lgbm_params, lgbm_score = random_search_tuner(
            LGBMClassifier(scale_pos_weight=scale_pos_weight, random_state=42, verbosity=-1),
            param_dist, X_train, y_train,
            cv=cv, n_iter=n_trials, scoring='recall'
        )
    
    print(f"   ✅ 최적 파라미터: {lgbm_params}")
    print(f"   ✅ CV 점수: {lgbm_score:.4f}")
    tuned_models.append(('lgbm', lgbm_best))
    
    print("\n🎉 모든 모델 튜닝 완료!\n")
    
    return tuned_models


####################################
# 🗳️ Voting Ensemble
####################################
def train_voting_ensemble(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    cv_strategy: Optional[str] = 'stratified_kfold',  # CV 전략
    tuning_strategy: Optional[str] = None,  # 튜닝 전략
    rf_weight: int = 1,
    xgb_weight: int = 2,
    lgbm_weight: int = 2,
    voting: str = 'soft',
    n_splits: int = 5,
    n_trials: int = 50
) -> VotingClassifier:
    """
    Voting Classifier 학습 (투표 기반 앙상블)
    
    Args:
        X_train, y_train: 훈련 데이터
        cv_strategy: CV 전략 ('stratified_kfold', 'kfold', None)
        tuning_strategy: 튜닝 전략 (None, 'optuna', 'grid_search', 'random_search')
        rf_weight, xgb_weight, lgbm_weight: 각 모델의 투표 가중치
        voting: 'soft' (확률 평균) or 'hard' (다수결)
        n_splits: CV 폴드 수
        n_trials: 튜닝 시도 횟수
    
    Returns:
        VotingClassifier: 학습된 모델
    """
    
    # 클래스 불균형 계산
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # CV splitter 생성
    cv_splitter = _get_cv_splitter(cv_strategy, n_splits)
    
    # 기본 모델 생성
    estimators = _create_base_models(
        scale_pos_weight=scale_pos_weight,
        X_train=X_train,
        y_train=y_train,
        tuning_strategy=tuning_strategy,
        cv=cv_splitter,
        n_trials=n_trials
    )

    # Voting 앙상블 생성 및 학습
    voting_model = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=[rf_weight, xgb_weight, lgbm_weight],
        n_jobs=-1
    )
    voting_model.fit(X_train, y_train)
    print("✅ 학습 완료!")
    
    return voting_model


####################################
# 📚 Stacking Ensemble
####################################
def train_stacking_ensemble(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    cv_strategy: Optional[str] = 'stratified_kfold',  # CV 전략
    tuning_strategy: Optional[str] = None,  # 튜닝 전략
    final_estimator: Optional[Any] = None,
    n_splits: int = 5,
    n_trials: int = 50
) -> StackingClassifier:
    """
    Stacking Classifier 학습 (스태킹 앙상블)
    
    💡 Stacking은 왜 CV가 필수?
    - 베이스 모델이 "본 적 없는" 데이터로 예측값 생성
    - 메타 모델이 이 예측값으로 학습 → 과적합 방지!
    
    Args:
        X_train, y_train: 훈련 데이터
        cv_strategy: CV 전략 ('stratified_kfold', 'kfold', None)
        tuning_strategy: 튜닝 전략 (None, 'optuna', 'grid_search', 'random_search')
        final_estimator: 메타 모델 (None이면 LogisticRegression)
        n_splits: CV 폴드 수
        n_trials: 튜닝 시도 횟수
    
    Returns:
        StackingClassifier: 학습된 모델
    """
    
    # 클래스 불균형 계산
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    # CV splitter 생성
    cv_splitter = _get_cv_splitter(cv_strategy, n_splits)
    
    # 기본 모델 생성
    estimators = _create_base_models(
        scale_pos_weight=scale_pos_weight,
        X_train=X_train,
        y_train=y_train,
        tuning_strategy=tuning_strategy,
        cv=cv_splitter,
        n_trials=n_trials
    )

    # 메타 모델 설정
    if final_estimator is None:
        final_estimator = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )

    # Stacking 앙상블 생성 및 학습
    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        stack_method='predict_proba',
        cv=cv_splitter,  # 👈 과적합 방지용 CV (필수!)
        n_jobs=-1
    )
    
    stacking_model.fit(X_train, y_train)
    print("✅ 학습 완료!")
    
    return stacking_model


####################################
# 📈 단일 모델: Logistic Regression
####################################
def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: str = 'balanced'
) -> LogisticRegression:
    """
    로지스틱 회귀 모델 학습
    
    💡 개선: max_iter를 5000으로 증가하여 수렴 경고 방지
    """
    model = LogisticRegression(
        max_iter=5000,  # 👈 ConvergenceWarning 방지
        random_state=42,
        class_weight=class_weight
    )
    model.fit(X_train, y_train)
    return model


####################################
# 📊 모델 평가 함수
####################################
def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fold_num: Optional[int] = None,
    n_splits: Optional[int] = None,
    print_report: bool = True
) -> Dict[str, float]:
    """
    모델을 평가하고 지표를 출력 및 반환합니다.
    
    💡 개선 사항:
    1. 평가 결과를 딕셔너리로 반환 → 저장/비교 가능
    2. print_report 옵션으로 출력 제어
    3. predict_proba 지원 여부 확인 (에러 방지)
    
    Args:
        model: 학습된 모델
        X_test: 테스트 데이터
        y_test: 테스트 타겟
        fold_num: 현재 폴드 번호 (교차검증 시)
        n_splits: 전체 폴드 수 (교차검증 시)
        print_report: 결과 출력 여부
    
    Returns:
        Dict[str, float]: 평가 지표들
        {
            'accuracy': 정확도,
            'roc_auc': ROC-AUC 점수,
            'f1': F1 점수,
            'recall': 재현율,
            'precision': 정밀도
        }
        
    예시:
        >>> metrics = evaluate_model(model, X_test, y_test)
        >>> print(f"F1 Score: {metrics['f1']:.4f}")
    """
    
    # 예측
    y_pred = model.predict(X_test)
    
    # ROC-AUC와 PR-AUC는 predict_proba를 지원하는 모델만 계산 가능
    roc_auc = None
    pr_auc = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)  # PR-AUC 계산
    
    # 각종 지표 계산
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc if roc_auc is not None else 0.0,
        'pr_auc': pr_auc if pr_auc is not None else 0.0,  # PR-AUC 추가
        'f1': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred)
    }
    
    # 출력 (필요한 경우에만)
    if print_report:
        if fold_num is not None and n_splits is not None:
            print(f"\n{'='*60}")
            print(f"  Fold {fold_num}/{n_splits} 평가 결과")
            print(f"{'='*60}")
        
        print(f"📊 정확도 (Accuracy):  {metrics['accuracy']:.4f}")
        if roc_auc is not None:
            print(f"📊 ROC-AUC:            {metrics['roc_auc']:.4f}")
        if pr_auc is not None:
            print(f"📊 PR-AUC:             {metrics['pr_auc']:.4f}")
        print(f"📊 F1 Score:           {metrics['f1']:.4f}")
        print(f"📊 재현율 (Recall):     {metrics['recall']:.4f}")
        print(f"📊 정밀도 (Precision):  {metrics['precision']:.4f}")
        print("\n📋 상세 분류 리포트:")
        print(classification_report(y_test, y_pred))
    
    return metrics

