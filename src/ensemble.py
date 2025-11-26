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


def _get_cv_splitter(
    cv_method: Optional[str] = None,
    n_splits: int = 5,
    shuffle: bool = True,
    random_state: int = 42
) -> Union[int, Any]:
    
    if cv_method is None or cv_method == 'default':
        return n_splits
    
    elif cv_method == 'stratified_kfold':
        return StratifiedKFold(
            n_splits=n_splits,
            shuffle=shuffle,
            random_state=random_state
        )
    
    elif cv_method == 'kfold':
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

def _create_base_models_from_params(
    best_params: Dict[str, Dict],
    scale_pos_weight: float
) -> List[Tuple[str, Any]]:

    estimators = []
    
    if 'rf' in best_params:
        rf_params = best_params['rf'].copy()
        rf_params.update({
            'class_weight': 'balanced',
            'random_state': 42,
            'n_jobs': -1
        })
        rf_model = RandomForestClassifier(**rf_params)
        estimators.append(('rf', rf_model))
    
    if 'xgb' in best_params:
        xgb_params = best_params['xgb'].copy()
        xgb_params.update({ 
            'scale_pos_weight': scale_pos_weight,
            'eval_metric': 'logloss',
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        })
        xgb_model = XGBClassifier(**xgb_params)
        estimators.append(('xgb', xgb_model))
    
    if 'lgbm' in best_params:
        lgbm_params = best_params['lgbm'].copy()
        lgbm_params.update({
            'scale_pos_weight': scale_pos_weight,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': -1
        })
        lgbm_model = LGBMClassifier(**lgbm_params)
        estimators.append(('lgbm', lgbm_model))
    
    return estimators

def _create_base_models(
    scale_pos_weight: float,
    X_train: Optional[pd.DataFrame] = None,
    y_train: Optional[pd.Series] = None,
    tuning_strategy: Optional[str] = None,
    cv: int = 5,
    n_trials: int = 50,
    return_params: bool = False
) -> Union[List[Tuple[str, Any]], Tuple[List[Tuple[str, Any]], Dict]]:

    
    if tuning_strategy is not None:
        if X_train is None or y_train is None:
            raise ValueError(
                "튜닝을 하려면 X_train과 y_train이 필요합니다\n"
                "   _create_base_models(..., X_train=X, y_train=y, tuning_strategy='optuna')"
            )
        print(f"\n 하이퍼파라미터 튜닝 모드: {tuning_strategy}")
        print(f"   데이터 크기: {X_train.shape}, CV={cv}, Trials={n_trials}")
        tuned_estimators, best_params = _tune_base_models(
            X_train=X_train,
            y_train=y_train,
            scale_pos_weight=scale_pos_weight,
            tuning_strategy=tuning_strategy,
            cv=cv,
            n_trials=n_trials
        )
        if return_params:
            return tuned_estimators, best_params
        return tuned_estimators
    
    print("\n 기본 파라미터 모드 (튜닝 없음)")
    
    rf_model = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42,
        n_jobs=-1
    )

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

    estimators = [
        ('rf', rf_model),
        ('xgb', xgb_model),
        ('lgbm', lgbm_model)
    ]
    
    default_params = {
        'rf': rf_model.get_params(),
        'xgb': xgb_model.get_params(),
        'lgbm': lgbm_model.get_params()
    }
    
    if return_params:
        return estimators, default_params
    return estimators

def _tune_base_models(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    scale_pos_weight: float,
    tuning_strategy: str = 'optuna',
    cv: int = 5,
    n_trials: int = 50
) -> Tuple[List[Tuple[str, Any]], Dict[str, Dict]]:

    from src.tuner import grid_search_tuner, random_search_tuner, optuna_tuner
    import optuna
    
    tuned_models = []
    best_params = {}
    
    print("\nRandom Forest 튜닝 중")
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
    else:
        raise ValueError(
            f"Unknown tuning_strategy: {tuning_strategy}. "
            "지원되는 값: 'optuna', 'grid_search', 'random_search'"
        )
    
    print(f" 최적 파라미터: {rf_params}")
    print(f" CV 점수: {rf_score:.4f}")
    tuned_models.append(('rf', rf_best))
    best_params['rf'] = rf_params
    
    print("\n XGBoost 튜닝 중")
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
    
    print(f" 최적 파라미터: {xgb_params}")
    print(f" CV 점수: {xgb_score:.4f}")
    tuned_models.append(('xgb', xgb_best))
    best_params['xgb'] = xgb_params
    
    print("\nLightGBM 튜닝 중")
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
    
    print(f" 최적 파라미터: {lgbm_params}")
    print(f" CV 점수: {lgbm_score:.4f}")
    tuned_models.append(('lgbm', lgbm_best))
    best_params['lgbm'] = lgbm_params
    
    print("\n모든 모델 튜닝 완료\n")
    
    return tuned_models, best_params

def train_voting_ensemble(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    cv_strategy: Optional[str] = 'stratified_kfold', 
    tuning_strategy: Optional[str] = None,  
    best_params: Optional[Dict] = None, 
    rf_weight: int = 1,
    xgb_weight: int = 2,
    lgbm_weight: int = 2,
    voting: str = 'soft',
    n_splits: int = 5,
    n_trials: int = 50,
    return_params: bool = False  
) -> Union[VotingClassifier, Tuple[VotingClassifier, Dict]]:
    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    cv_splitter = _get_cv_splitter(cv_strategy, n_splits)
    
    if best_params is not None:
        print("\n 튜닝된 파라미터 재사용")
        estimators = _create_base_models_from_params(
            best_params=best_params,
            scale_pos_weight=scale_pos_weight
        )
        extracted_params = best_params
    elif return_params:
        estimators, extracted_params = _create_base_models(
            scale_pos_weight=scale_pos_weight,
            X_train=X_train,
            y_train=y_train,
            tuning_strategy=tuning_strategy,
            cv=cv_splitter,
            n_trials=n_trials,
            return_params=True
        )
    else:
        estimators = _create_base_models(
            scale_pos_weight=scale_pos_weight,
            X_train=X_train,
            y_train=y_train,
            tuning_strategy=tuning_strategy,
            cv=cv_splitter,
            n_trials=n_trials
        )
        extracted_params = None

    voting_model = VotingClassifier(
        estimators=estimators,
        voting=voting,
        weights=[rf_weight, xgb_weight, lgbm_weight],
        n_jobs=-1
    )
    voting_model.fit(X_train, y_train)
    print("학습 완료")
    
    if return_params:
        return voting_model, extracted_params
    return voting_model

def train_stacking_ensemble(
    X_train: pd.DataFrame, 
    y_train: pd.Series, 
    cv_strategy: Optional[str] = 'stratified_kfold',  
    tuning_strategy: Optional[str] = None, 
    best_params: Optional[Dict] = None,  
    final_estimator: Optional[Any] = None,
    n_splits: int = 5,
    n_trials: int = 50,
    return_params: bool = False 
) -> Union[StackingClassifier, Tuple[StackingClassifier, Dict]]:

    
    scale_pos_weight = (y_train == 0).sum() / (y_train == 1).sum()

    cv_splitter = _get_cv_splitter(cv_strategy, n_splits)
    
    if best_params is not None:
        print("\n 튜닝된 파라미터 재사용")
        estimators = _create_base_models_from_params(
            best_params=best_params,
            scale_pos_weight=scale_pos_weight
        )
        extracted_params = best_params
    elif return_params:
        estimators, extracted_params = _create_base_models(
            scale_pos_weight=scale_pos_weight,
            X_train=X_train,
            y_train=y_train,
            tuning_strategy=tuning_strategy,
            cv=cv_splitter,
            n_trials=n_trials,
            return_params=True
        )
    else:
        estimators = _create_base_models(
            scale_pos_weight=scale_pos_weight,
            X_train=X_train,
            y_train=y_train,
            tuning_strategy=tuning_strategy,
            cv=cv_splitter,
            n_trials=n_trials
        )
        extracted_params = None

    if final_estimator is None:
        final_estimator = LogisticRegression(
            max_iter=1000,
            random_state=42,
            class_weight='balanced'
        )

    stacking_model = StackingClassifier(
        estimators=estimators,
        final_estimator=final_estimator,
        stack_method='predict_proba',
        cv=cv_splitter,  
        n_jobs=-1
    )
    
    stacking_model.fit(X_train, y_train)
    print("학습 완료")
    
    if return_params:
        return stacking_model, extracted_params
    return stacking_model

def train_logistic_regression(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    class_weight: str = 'balanced'
) -> LogisticRegression:

    model = LogisticRegression(
        max_iter=5000, 
        random_state=42,
        class_weight=class_weight
    )
    model.fit(X_train, y_train)
    return model

def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    fold_num: Optional[int] = None,
    n_splits: Optional[int] = None,
    print_report: bool = True
) -> Dict[str, float]:

    y_pred = model.predict(X_test)
    
    roc_auc = None
    pr_auc = None
    if hasattr(model, "predict_proba"):
        y_proba = model.predict_proba(X_test)[:, 1]
        roc_auc = roc_auc_score(y_test, y_proba)
        pr_auc = average_precision_score(y_test, y_proba)  
    
    metrics = {
        'accuracy': accuracy_score(y_test, y_pred),
        'roc_auc': roc_auc if roc_auc is not None else 0.0,
        'pr_auc': pr_auc if pr_auc is not None else 0.0,  
        'f1': f1_score(y_test, y_pred),
        'recall': recall_score(y_test, y_pred),
        'precision': precision_score(y_test, y_pred)
    }
    
    if print_report:
        if fold_num is not None and n_splits is not None:
            print(f"\n{'='*60}")
            print(f"  Fold {fold_num}/{n_splits} 평가 결과")
            print(f"{'='*60}")
        
        print(f" 정확도 (Accuracy):  {metrics['accuracy']:.4f}")
        if roc_auc is not None:
            print(f" ROC-AUC:            {metrics['roc_auc']:.4f}")
        if pr_auc is not None:
            print(f" PR-AUC:             {metrics['pr_auc']:.4f}")
        print(f" F1 Score:           {metrics['f1']:.4f}")
        print(f" 재현율 (Recall):     {metrics['recall']:.4f}")
        print(f" 정밀도 (Precision):  {metrics['precision']:.4f}")
        print("\n 상세 분류 리포트:")
        print(classification_report(y_test, y_pred))
    
    return metrics