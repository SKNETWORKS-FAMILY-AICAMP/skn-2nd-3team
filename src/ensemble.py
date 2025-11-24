"""
앙상블 전략 구현
- Voting Classifier (Hard/Soft)
- Stacking
- Blending
- Weighted Average
"""
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report
import numpy as np
from .models import get_base_models, get_scaled_models, train_model, predict_model


def voting_ensemble(base_models, X_train, y_train, voting='soft'):
    """
    Voting Classifier 앙상블
    
    Args:
        base_models: 기본 모델들의 딕셔너리
        X_train: 학습 데이터
        y_train: 타겟 변수
        voting: 'hard' 또는 'soft' (기본값: 'soft')
    
    Returns:
        학습된 Voting Classifier
    """
    # Voting에 사용할 모델들 준비
    estimators = []
    scaled_models = get_scaled_models()
    
    for name, model in base_models.items():
        if name in scaled_models:
            # 스케일링이 필요한 모델은 별도 처리 필요
            # 여기서는 간단히 스케일링 없이 사용
            estimators.append((name, model))
        else:
            estimators.append((name, model))
    
    voting_clf = VotingClassifier(
        estimators=estimators,
        voting=voting,
        n_jobs=-1
    )
    
    voting_clf.fit(X_train, y_train)
    return voting_clf


def stacking_ensemble(base_models, X_train, y_train, X_val, y_val, meta_model=None):
    """
    Stacking 앙상블
    
    Args:
        base_models: 기본 모델들의 딕셔너리
        X_train: 학습 데이터
        y_train: 타겟 변수
        X_val: 검증 데이터 (meta-learner 학습용)
        y_val: 검증 타겟 변수
        meta_model: Meta-learner (기본값: LogisticRegression)
    
    Returns:
        학습된 base 모델들과 meta-learner
    """
    if meta_model is None:
        meta_model = LogisticRegression(random_state=42)
    
    # 1단계: Base 모델들 학습
    trained_models = {}
    scaled_models = get_scaled_models()
    
    for name, model in base_models.items():
        scale = name in scaled_models
        trained_model, scaler = train_model(model, X_train, y_train, scale=scale)
        trained_models[name] = {'model': trained_model, 'scaler': scaler}
    
    # 2단계: Base 모델들의 예측을 feature로 사용
    meta_features = []
    for name, model_dict in trained_models.items():
        pred_proba = predict_model(
            model_dict['model'], 
            X_val, 
            scaler=model_dict['scaler']
        )
        meta_features.append(pred_proba)
    
    meta_features = np.column_stack(meta_features)
    
    # 3단계: Meta-learner 학습
    meta_model.fit(meta_features, y_val)
    
    return trained_models, meta_model


def blending_ensemble(base_models, X_train, y_train, X_val, y_val, weights=None):
    """
    Blending 앙상블 (가중 평균)
    
    Args:
        base_models: 기본 모델들의 딕셔너리
        X_train: 학습 데이터
        y_train: 타겟 변수
        X_val: 검증 데이터
        y_val: 검증 타겟 변수
        weights: 모델별 가중치 (기본값: 균등 가중치)
    
    Returns:
        학습된 모델들과 가중치
    """
    # Base 모델들 학습
    trained_models = {}
    scaled_models = get_scaled_models()
    
    for name, model in base_models.items():
        scale = name in scaled_models
        trained_model, scaler = train_model(model, X_train, y_train, scale=scale)
        trained_models[name] = {'model': trained_model, 'scaler': scaler}
    
    # 가중치 설정 (없으면 균등 가중치)
    if weights is None:
        weights = {name: 1.0 / len(trained_models) for name in trained_models.keys()}
    
    return trained_models, weights


def predict_stacking(trained_models, meta_model, X):
    """
    Stacking 모델 예측
    
    Args:
        trained_models: 학습된 base 모델들
        meta_model: 학습된 meta-learner
        X: 예측할 데이터
    
    Returns:
        예측 확률
    """
    meta_features = []
    for name, model_dict in trained_models.items():
        pred_proba = predict_model(
            model_dict['model'],
            X,
            scaler=model_dict['scaler']
        )
        meta_features.append(pred_proba)
    
    meta_features = np.column_stack(meta_features)
    return meta_model.predict_proba(meta_features)[:, 1]


def predict_blending(trained_models, weights, X):
    """
    Blending 모델 예측
    
    Args:
        trained_models: 학습된 모델들
        weights: 모델별 가중치
        X: 예측할 데이터
    
    Returns:
        예측 확률 (가중 평균)
    """
    predictions = []
    for name, model_dict in trained_models.items():
        pred_proba = predict_model(
            model_dict['model'],
            X,
            scaler=model_dict['scaler']
        )
        weighted_pred = pred_proba * weights[name]
        predictions.append(weighted_pred)
    
    return np.sum(predictions, axis=0)


def evaluate_model(model, X_test, y_test, model_type='single'):
    """
    모델 예측 및 평가 결과 출력
    
    Args:
        model: 학습된 모델 또는 (trained_models, meta_model/weights) 튜플
        X_test: 테스트 데이터
        y_test: 테스트 타겟 변수
        model_type: 'single', 'stacking', 'blending', 'voting'
    """
    if model_type == 'single':
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    elif model_type == 'voting':
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]
    elif model_type == 'stacking':
        trained_models, meta_model = model
        y_proba = predict_stacking(trained_models, meta_model, X_test)
        y_pred = (y_proba >= 0.5).astype(int)
    elif model_type == 'blending':
        trained_models, weights = model
        y_proba = predict_blending(trained_models, weights, X_test)
        y_pred = (y_proba >= 0.5).astype(int)
    else:
        raise ValueError(f"Unknown model_type: {model_type}")
    
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    
    print(f"정확도: {acc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred))
    
    return {
        'accuracy': acc,
        'roc_auc': roc,
        'y_pred': y_pred,
        'y_proba': y_proba
    }


# 하위 호환성을 위한 함수들
def train_logistic_regression(X_train, y_train):
    """로지스틱 회귀 모델 학습 (하위 호환성)"""
    from .models import get_base_models
    models = get_base_models()
    model = models['LogisticRegression']
    model.fit(X_train, y_train)
    return model


def create_voting_ensemble_pipeline(preprocessor, rf_params=None, xgb_params=None, weights=[1, 2], voting='soft'):
    """
    Pipeline 기반 Voting Ensemble 생성 (노트북 기반)
    
    Args:
        preprocessor: 전처리 파이프라인
        rf_params: RandomForest 파라미터 딕셔너리 (None이면 최적 파라미터 사용)
        xgb_params: XGBoost 파라미터 딕셔너리 (None이면 최적 파라미터 사용)
        weights: 모델별 가중치
        voting: 'hard' 또는 'soft'
        
    Returns:
        Voting Ensemble Pipeline
    """
    from sklearn.ensemble import RandomForestClassifier, VotingClassifier
    from sklearn.pipeline import Pipeline
    
    try:
        from xgboost import XGBClassifier
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False
    
    # RandomForest 파라미터 설정
    if rf_params is None:
        rf_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'class_weight': 'balanced',
            'random_state': 42
        }
    
    best_rf = RandomForestClassifier(**rf_params)
    
    # XGBoost 파라미터 설정
    if XGBOOST_AVAILABLE:
        if xgb_params is None:
            xgb_params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 5,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'eval_metric': 'logloss',
                'random_state': 42
            }
        
        best_xgb = XGBClassifier(**xgb_params)
        
        voting_model = Pipeline([
            ('preprocess', preprocessor),
            ('ensemble', VotingClassifier(
                estimators=[
                    ('rf', best_rf),
                    ('xgb', best_xgb)
                ],
                voting=voting,
                weights=weights
            ))
        ])
    else:
        # XGBoost가 없으면 RF만 사용
        voting_model = Pipeline([
            ('preprocess', preprocessor),
            ('ensemble', best_rf)
        ])
    
    return voting_model


def create_stacking_ensemble_pipeline(preprocessor, rf_params=None, xgb_params=None, meta_model=None, cv=5):
    """
    Pipeline 기반 Stacking Ensemble 생성 (노트북 기반)
    
    Args:
        preprocessor: 전처리 파이프라인
        rf_params: RandomForest 파라미터 딕셔너리 (None이면 최적 파라미터 사용)
        xgb_params: XGBoost 파라미터 딕셔너리 (None이면 최적 파라미터 사용)
        meta_model: Meta-learner (None이면 LogisticRegression 사용)
        cv: 교차 검증 폴드 수
        
    Returns:
        Stacking Ensemble Pipeline
    """
    from sklearn.ensemble import RandomForestClassifier, StackingClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import Pipeline
    
    try:
        from xgboost import XGBClassifier
        XGBOOST_AVAILABLE = True
    except ImportError:
        XGBOOST_AVAILABLE = False
    
    # RandomForest 파라미터 설정
    if rf_params is None:
        rf_params = {
            'n_estimators': 200,
            'max_depth': 10,
            'min_samples_split': 5,
            'class_weight': 'balanced',
            'random_state': 42
        }
    
    best_rf = RandomForestClassifier(**rf_params)
    
    # Meta-learner 설정
    if meta_model is None:
        meta_model = LogisticRegression(max_iter=1000, random_state=42)
    
    if XGBOOST_AVAILABLE:
        # XGBoost 파라미터 설정
        if xgb_params is None:
            xgb_params = {
                'n_estimators': 200,
                'learning_rate': 0.05,
                'max_depth': 5,
                'subsample': 0.9,
                'colsample_bytree': 0.9,
                'eval_metric': 'logloss',
                'random_state': 42
            }
        
        best_xgb = XGBClassifier(**xgb_params)
        
        stacking_model = Pipeline([
            ('preprocess', preprocessor),
            ('stack', StackingClassifier(
                estimators=[
                    ('rf', best_rf),
                    ('xgb', best_xgb)
                ],
                final_estimator=meta_model,
                stack_method='predict_proba',
                cv=cv,
                n_jobs=-1
            ))
        ])
    else:
        # XGBoost가 없으면 RF만 사용
        stacking_model = Pipeline([
            ('preprocess', preprocessor),
            ('stack', StackingClassifier(
                estimators=[('rf', best_rf)],
                final_estimator=meta_model,
                stack_method='predict_proba',
                cv=cv,
                n_jobs=-1
            ))
        ])
    
    return stacking_model