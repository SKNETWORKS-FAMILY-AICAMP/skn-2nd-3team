"""
Churn 예측을 위한 다양한 머신러닝 모델들
"""
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import StandardScaler
import numpy as np

try:
    from xgboost import XGBClassifier
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    XGBClassifier = None

try:
    from lightgbm import LGBMClassifier
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False
    LGBMClassifier = None

try:
    from catboost import CatBoostClassifier
    CATBOOST_AVAILABLE = True
except ImportError:
    CATBOOST_AVAILABLE = False
    CatBoostClassifier = None


def get_base_models(random_state=42):
    """
    Churn 예측에 적합한 기본 모델들을 반환
    
    Returns:
        dict: 모델 이름과 모델 객체의 딕셔너리
    """
    models = {}
    
    # 1. Logistic Regression - 해석 가능성과 안정성
    models['LogisticRegression'] = LogisticRegression(
        max_iter=1000,
        random_state=random_state,
        solver='lbfgs'
    )
    
    # 2. Random Forest - 비선형 관계 포착, 변수 중요도 제공
    models['RandomForest'] = RandomForestClassifier(
        n_estimators=100,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=random_state,
        n_jobs=-1
    )
    
    # 3. Gradient Boosting - 순차적 오차 보정
    models['GradientBoosting'] = GradientBoostingClassifier(
        n_estimators=100,
        learning_rate=0.1,
        max_depth=5,
        random_state=random_state
    )
    
    # 4. XGBoost - 성능 우수, 정규화 포함
    if XGBOOST_AVAILABLE:
        models['XGBoost'] = XGBClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    # 5. LightGBM - 빠른 학습 속도, 범주형 변수 처리 우수
    if LIGHTGBM_AVAILABLE:
        models['LightGBM'] = LGBMClassifier(
            n_estimators=100,
            learning_rate=0.1,
            max_depth=5,
            random_state=random_state,
            verbose=-1
        )
    
    # 6. CatBoost - 범주형 변수 자동 처리, 과적합 방지
    if CATBOOST_AVAILABLE:
        models['CatBoost'] = CatBoostClassifier(
            iterations=100,
            learning_rate=0.1,
            depth=5,
            random_state=random_state,
            verbose=False
        )
    
    # 7. SVM - 선형/비선형 경계 학습 (스케일링 필요)
    models['SVM'] = SVC(
        kernel='rbf',
        probability=True,
        random_state=random_state
    )
    
    # 8. Neural Network - 복잡한 비선형 패턴 학습 (스케일링 필요)
    models['NeuralNetwork'] = MLPClassifier(
        hidden_layer_sizes=(100, 50),
        max_iter=500,
        random_state=random_state,
        early_stopping=True
    )
    
    return models


def get_optimized_models(random_state=42):
    """
    GridSearch로 찾은 최적 파라미터를 가진 모델들 반환 (노트북 기반)
    
    Returns:
        dict: 모델 이름과 최적화된 모델 객체의 딕셔너리
    """
    models = {}
    
    # GridSearch 결과 기반 최적 파라미터
    # RandomForest: {'class_weight': 'balanced', 'max_depth': None, 'min_samples_split': 2, 'n_estimators': 300}
    models['RandomForest_Optimized'] = RandomForestClassifier(
        n_estimators=300,
        max_depth=None,
        min_samples_split=2,
        class_weight='balanced',
        random_state=random_state,
        n_jobs=-1
    )
    
    # XGBoost 최적 파라미터 (SMOTE 없이 사용할 경우)
    if XGBOOST_AVAILABLE:
        models['XGBoost_Optimized'] = XGBClassifier(
            n_estimators=200,
            learning_rate=0.1,
            max_depth=5,
            subsample=0.9,
            colsample_bytree=0.9,
            random_state=random_state,
            eval_metric='logloss',
            use_label_encoder=False
        )
    
    return models


def get_scaled_models():
    """
    스케일링이 필요한 모델들의 리스트 반환
    """
    return ['SVM', 'NeuralNetwork', 'LogisticRegression']


def train_model(model, X_train, y_train, scale=False):
    """
    모델 학습
    
    Args:
        model: 학습할 모델 객체
        X_train: 학습 데이터
        y_train: 타겟 변수
        scale: 스케일링 여부
    
    Returns:
        학습된 모델과 스케일러(사용한 경우)
    """
    scaler = None
    if scale:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
        model.fit(X_train_scaled, y_train)
    else:
        model.fit(X_train, y_train)
    
    return model, scaler


def predict_model(model, X, scaler=None):
    """
    모델 예측
    
    Args:
        model: 학습된 모델
        X: 예측할 데이터
        scaler: 스케일러 (사용한 경우)
    
    Returns:
        예측 확률 (positive class)
    """
    if scaler is not None:
        X = scaler.transform(X)
    
    return model.predict_proba(X)[:, 1]

