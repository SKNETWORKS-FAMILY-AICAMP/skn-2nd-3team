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
