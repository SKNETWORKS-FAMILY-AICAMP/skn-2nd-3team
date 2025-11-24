"""
Model Selection 파이프라인
- 다양한 모델 학습 및 평가
- 교차 검증을 통한 성능 비교
- 최적 모델 선택
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import StratifiedKFold, cross_val_score, cross_validate
from sklearn.metrics import accuracy_score, roc_auc_score, precision_score, recall_score, f1_score, average_precision_score
from sklearn.preprocessing import StandardScaler
from .models import get_base_models, get_scaled_models, train_model, predict_model
from .ensemble import voting_ensemble, stacking_ensemble, blending_ensemble
import warnings
warnings.filterwarnings('ignore')

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False


def evaluate_single_model(model, X_train, y_train, X_test, y_test, model_name, cv=5):
    """
    단일 모델 평가
    
    Args:
        model: 학습할 모델
        X_train: 학습 데이터
        y_train: 학습 타겟
        X_test: 테스트 데이터
        y_test: 테스트 타겟
        model_name: 모델 이름
        cv: 교차 검증 폴드 수
    
    Returns:
        평가 결과 딕셔너리
    """
    scaled_models = get_scaled_models()
    scale = model_name in scaled_models
    
    # 교차 검증 (ROC-AUC와 PR-AUC 모두 계산)
    cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    scoring = {
        'roc_auc': 'roc_auc',
        'pr_auc': 'average_precision'
    }
    
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        cv_results = cross_validate(
            model, X_train_scaled, y_train,
            cv=cv_fold,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        model.fit(X_train_scaled, y_train)
        X_test_scaled = scaler.transform(X_test)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        cv_results = cross_validate(
            model, X_train, y_train,
            cv=cv_fold,
            scoring=scoring,
            n_jobs=-1,
            return_train_score=False
        )
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    
    # 교차 검증 결과 추출
    cv_roc_scores = cv_results['test_roc_auc']
    cv_pr_scores = cv_results['test_pr_auc']
    
    # 테스트 세트 평가
    test_acc = accuracy_score(y_test, y_pred)
    test_roc = roc_auc_score(y_test, y_proba)
    test_pr_auc = average_precision_score(y_test, y_proba)
    test_precision = precision_score(y_test, y_pred)
    test_recall = recall_score(y_test, y_pred)
    test_f1 = f1_score(y_test, y_pred)
    
    return {
        'model_name': model_name,
        'cv_roc_mean': cv_roc_scores.mean(),
        'cv_roc_std': cv_roc_scores.std(),
        'cv_pr_mean': cv_pr_scores.mean(),
        'cv_pr_std': cv_pr_scores.std(),
        'cv_mean': (cv_roc_scores.mean() + cv_pr_scores.mean()) / 2,  # Combined for backward compatibility
        'cv_std': (cv_roc_scores.std() + cv_pr_scores.std()) / 2,  # Combined for backward compatibility
        'test_accuracy': test_acc,
        'test_roc_auc': test_roc,
        'test_pr_auc': test_pr_auc,
        'test_precision': test_precision,
        'test_recall': test_recall,
        'test_f1': test_f1,
        'model': model,
        'scaler': scaler if scale else None
    }


def compare_all_models(X_train, y_train, X_test, y_test, cv=5):
    """
    모든 기본 모델 비교
    
    Args:
        X_train: 학습 데이터
        y_train: 학습 타겟
        X_test: 테스트 데이터
        y_test: 테스트 타겟
        cv: 교차 검증 폴드 수
    
    Returns:
        결과 DataFrame
    """
    base_models = get_base_models()
    results = []
    
    print("=" * 80)
    print("모델 성능 비교 시작")
    print("=" * 80)
    
    for name, model in base_models.items():
        print(f"\n[{name}] 학습 중...")
        try:
            result = evaluate_single_model(
                model, X_train, y_train, X_test, y_test, name, cv
            )
            results.append(result)
            print(f"  CV ROC-AUC: {result['cv_roc_mean']:.4f} (+/- {result['cv_roc_std']:.4f})")
            print(f"  CV PR-AUC: {result['cv_pr_mean']:.4f} (+/- {result['cv_pr_std']:.4f})")
            print(f"  Test ROC-AUC: {result['test_roc_auc']:.4f}")
            print(f"  Test PR-AUC: {result['test_pr_auc']:.4f}")
        except Exception as e:
            print(f"  오류 발생: {str(e)}")
            continue
    
    # 결과를 DataFrame으로 변환
    df_results = pd.DataFrame([
        {
            'Model': r['model_name'],
            'CV_ROC_AUC_Mean': r['cv_roc_mean'],
            'CV_ROC_AUC_Std': r['cv_roc_std'],
            'CV_PR_AUC_Mean': r['cv_pr_mean'],
            'CV_PR_AUC_Std': r['cv_pr_std'],
            'Test_Accuracy': r['test_accuracy'],
            'Test_ROC_AUC': r['test_roc_auc'],
            'Test_PR_AUC': r['test_pr_auc'],
            'Test_Precision': r['test_precision'],
            'Test_Recall': r['test_recall'],
            'Test_F1': r['test_f1']
        }
        for r in results
    ])
    
    # ROC-AUC와 PR-AUC의 평균 기준으로 정렬 (둘 다 고려)
    df_results['Test_Combined_AUC'] = (df_results['Test_ROC_AUC'] + df_results['Test_PR_AUC']) / 2
    df_results['CV_Combined_AUC'] = (df_results['CV_ROC_AUC_Mean'] + df_results['CV_PR_AUC_Mean']) / 2
    df_results = df_results.sort_values('Test_Combined_AUC', ascending=False)
    
    print("\n" + "=" * 80)
    print("모델 성능 비교 결과")
    print("=" * 80)
    print(df_results.to_string(index=False))
    
    return df_results, results


def compare_ensemble_models(X_train, y_train, X_val, y_val, X_test, y_test):
    """
    앙상블 모델들 비교
    
    Args:
        X_train: 학습 데이터
        y_train: 학습 타겟
        X_val: 검증 데이터
        y_val: 검증 타겟
        X_test: 테스트 데이터
        y_test: 테스트 타겟
    
    Returns:
        결과 딕셔너리
    """
    base_models = get_base_models()
    results = {}
    
    print("\n" + "=" * 80)
    print("앙상블 모델 성능 비교")
    print("=" * 80)
    
    # 1. Voting Classifier (Soft)
    print("\n[Voting Classifier (Soft)] 학습 중...")
    try:
        voting_model = voting_ensemble(base_models, X_train, y_train, voting='soft')
        y_pred = voting_model.predict(X_test)
        y_proba = voting_model.predict_proba(X_test)[:, 1]
        
        results['Voting_Soft'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'model': voting_model
        }
        print(f"  Test ROC-AUC: {results['Voting_Soft']['roc_auc']:.4f}")
        print(f"  Test PR-AUC: {results['Voting_Soft']['pr_auc']:.4f}")
    except Exception as e:
        print(f"  오류 발생: {str(e)}")
    
    # 2. Stacking
    print("\n[Stacking] 학습 중...")
    try:
        stacking_models = stacking_ensemble(
            base_models, X_train, y_train, X_val, y_val
        )
        from .ensemble import predict_stacking
        trained_models, meta_model = stacking_models
        y_proba = predict_stacking(trained_models, meta_model, X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        
        results['Stacking'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'model': stacking_models
        }
        print(f"  Test ROC-AUC: {results['Stacking']['roc_auc']:.4f}")
        print(f"  Test PR-AUC: {results['Stacking']['pr_auc']:.4f}")
    except Exception as e:
        print(f"  오류 발생: {str(e)}")
    
    # 3. Blending
    print("\n[Blending] 학습 중...")
    try:
        blending_models = blending_ensemble(
            base_models, X_train, y_train, X_val, y_val
        )
        from .ensemble import predict_blending
        trained_models, weights = blending_models
        y_proba = predict_blending(trained_models, weights, X_test)
        y_pred = (y_proba >= 0.5).astype(int)
        
        results['Blending'] = {
            'accuracy': accuracy_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'pr_auc': average_precision_score(y_test, y_proba),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1': f1_score(y_test, y_pred),
            'model': blending_models
        }
        print(f"  Test ROC-AUC: {results['Blending']['roc_auc']:.4f}")
        print(f"  Test PR-AUC: {results['Blending']['pr_auc']:.4f}")
    except Exception as e:
        print(f"  오류 발생: {str(e)}")
    
    # 결과 요약
    if results:
        print("\n" + "=" * 80)
        print("앙상블 모델 성능 비교 결과")
        print("=" * 80)
        ensemble_df = pd.DataFrame([
            {
                'Model': name,
                'Accuracy': r['accuracy'],
                'ROC_AUC': r['roc_auc'],
                'PR_AUC': r['pr_auc'],
                'Precision': r['precision'],
                'Recall': r['recall'],
                'F1': r['f1']
            }
            for name, r in results.items()
        ])
        # ROC-AUC와 PR-AUC의 평균 기준으로 정렬 (둘 다 고려)
        ensemble_df['Combined_AUC'] = (ensemble_df['ROC_AUC'] + ensemble_df['PR_AUC']) / 2
        ensemble_df = ensemble_df.sort_values('Combined_AUC', ascending=False)
        print(ensemble_df.to_string(index=False))
    
    return results


def select_best_model(base_results, ensemble_results=None, metric='combined_auc'):
    """
    최적 모델 선택
    
    Args:
        base_results: 기본 모델 결과 리스트
        ensemble_results: 앙상블 모델 결과 딕셔너리
        metric: 선택 기준 지표 ('roc_auc', 'pr_auc', 'combined_auc', 'accuracy', 'f1')
                'combined_auc'는 ROC-AUC와 PR-AUC의 평균
    
    Returns:
        최적 모델 정보
    """
    all_results = []
    
    # 기본 모델 결과 추가
    for r in base_results:
        if metric == 'combined_auc':
            score = (r['test_roc_auc'] + r['test_pr_auc']) / 2
        elif metric == 'roc_auc':
            score = r['test_roc_auc']
        elif metric == 'pr_auc':
            score = r['test_pr_auc']
        else:
            score = r[f'test_{metric}']
        
        all_results.append({
            'name': r['model_name'],
            'score': score,
            'roc_auc': r['test_roc_auc'],
            'pr_auc': r['test_pr_auc'],
            'model': r['model'],
            'scaler': r.get('scaler'),
            'type': 'base'
        })
    
    # 앙상블 모델 결과 추가
    if ensemble_results:
        for name, r in ensemble_results.items():
            if metric == 'combined_auc':
                score = (r['roc_auc'] + r['pr_auc']) / 2
            elif metric == 'roc_auc':
                score = r['roc_auc']
            elif metric == 'pr_auc':
                score = r['pr_auc']
            else:
                score = r[metric]
            
            all_results.append({
                'name': name,
                'score': score,
                'roc_auc': r['roc_auc'],
                'pr_auc': r['pr_auc'],
                'model': r['model'],
                'scaler': None,
                'type': 'ensemble'
            })
    
    # 최고 성능 모델 선택
    best = max(all_results, key=lambda x: x['score'])
    
    print("\n" + "=" * 80)
    if metric == 'combined_auc':
        print("최적 모델 선택 (기준: Combined AUC = (ROC-AUC + PR-AUC) / 2)")
    else:
        print(f"최적 모델 선택 (기준: {metric.upper()})")
    print("=" * 80)
    print(f"모델: {best['name']} ({best['type']})")
    print(f"ROC-AUC: {best['roc_auc']:.4f}")
    print(f"PR-AUC: {best['pr_auc']:.4f}")
    if metric == 'combined_auc':
        print(f"Combined AUC: {best['score']:.4f}")
    else:
        print(f"{metric.upper()}: {best['score']:.4f}")
    
    return best


def grid_search_rf(preprocessor, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0):
    """
    RandomForest GridSearchCV (노트북 기반)
    
    Args:
        preprocessor: 전처리 파이프라인
        X_train: 학습 데이터
        y_train: 학습 타겟
        cv: 교차 검증 폴드 수
        scoring: 평가 지표
        n_jobs: 병렬 처리 수
        verbose: 출력 상세도
        
    Returns:
        GridSearchCV 객체
    """
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from sklearn.pipeline import Pipeline
    
    rf_model = Pipeline([
        ('preprocessor', preprocessor),
        ('rf', RandomForestClassifier(random_state=42))
    ])
    
    rf_params = {
        'rf__n_estimators': [100, 200, 300],
        'rf__max_depth': [5, 10, 15, None],
        'rf__min_samples_split': [2, 5, 10],
        'rf__class_weight': ['balanced', None]
    }
    
    cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    rf_grid = GridSearchCV(
        estimator=rf_model,
        param_grid=rf_params,
        cv=cv_fold,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    rf_grid.fit(X_train, y_train)
    
    return rf_grid


def grid_search_xgb_smote(preprocessor, X_train, y_train, cv=5, scoring='roc_auc', n_jobs=-1, verbose=0):
    """
    XGBoost + SMOTE GridSearchCV (노트북 기반)
    
    Args:
        preprocessor: 전처리 파이프라인
        X_train: 학습 데이터
        y_train: 학습 타겟
        cv: 교차 검증 폴드 수
        scoring: 평가 지표
        n_jobs: 병렬 처리 수
        verbose: 출력 상세도
        
    Returns:
        GridSearchCV 객체
    """
    try:
        from imblearn.over_sampling import SMOTE
        from imblearn.pipeline import Pipeline as ImbPipeline
    except ImportError:
        raise ImportError("imbalanced-learn이 필요합니다. pip install imbalanced-learn")
    
    if not XGBOOST_AVAILABLE:
        raise ImportError("XGBoost가 필요합니다. pip install xgboost")
    
    from sklearn.model_selection import GridSearchCV, StratifiedKFold
    from xgboost import XGBClassifier
    
    smote_xgb_model = ImbPipeline([
        ('preprocessor', preprocessor),
        ('smote', SMOTE(random_state=42, sampling_strategy=0.8)),
        ('xgb', XGBClassifier(random_state=42, eval_metric='logloss'))
    ])
    
    xgb_params = {
        'xgb__n_estimators': [100, 200],
        'xgb__learning_rate': [0.05, 0.1],
        'xgb__max_depth': [5, 7],
        'smote__k_neighbors': [3, 5]
    }
    
    cv_fold = StratifiedKFold(n_splits=cv, shuffle=True, random_state=42)
    
    xgb_grid = GridSearchCV(
        estimator=smote_xgb_model,
        param_grid=xgb_params,
        cv=cv_fold,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose
    )
    
    xgb_grid.fit(X_train, y_train)
    
    return xgb_grid
