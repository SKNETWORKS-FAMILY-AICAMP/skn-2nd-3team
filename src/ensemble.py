from sklearn.metrics import accuracy_score, roc_auc_score, classification_report, f1_score, recall_score, precision_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import StackingClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
import numpy as np

####################################
def train_voting_ensemble(
    X_train, 
    y_train, 
    rf_weights=1, 
    xgb_weights=2
    ):
    """
    Random Forest와 XGBoost를 결합한 소프트 투표 앙상블 모델을 학습시키는 함수.

    Args:
        X_train (pd.DataFrame): 훈련용 특징 데이터.
        y_train (pd.Series): 훈련용 타겟 변수.
        preprocessor (ColumnTransformer): 전처리 파이프라인 객체.
        rf_weights (int): Random Forest 모델에 할당할 투표 가중치.
        xgb_weights (int): XGBoost 모델에 할당할 투표 가중치.

    Returns:
        Pipeline: 학습이 완료된 앙상블 파이프라인 객체 (voting_model).
    """
    
    scale_pos_weight_value = sum(y_train == 0) / sum(y_train == 1)

    best_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )

    best_xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight_value,
        eval_metric="logloss",
        random_state=42
    )

    voting_model =  VotingClassifier(
            estimators=[
                ('rf', best_rf),
                ('xgb', best_xgb)
            ],
            voting='soft',   
            weights=[rf_weights, xgb_weights], 
            n_jobs=-1 # 앙상블 학습 병렬 처리
        )

    voting_model.fit(X_train, y_train)
    
    return voting_model

########################################


def train_stacking_ensemble(
    X_train, 
    y_train, 
    cv_folds=5
    ):
    """
    Random Forest와 XGBoost를 기반으로 한 Stacking 앙상블 모델을 학습시키는 함수.

    Args:
        X_train (pd.DataFrame): 훈련용 특징 데이터.
        y_train (pd.Series): 훈련용 타겟 변수.
        preprocessor (ColumnTransformer): 전처리 파이프라인 객체.
        cv_folds (int): StackingClassifier 내부 교차 검증에 사용할 폴드 수.

    Returns:
        Pipeline: 학습이 완료된 Stacking 파이프라인 객체 (stacking_model).
    """
    
    scale_pos_weight_value = sum(y_train == 0) / sum(y_train == 1)

    best_rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        class_weight='balanced',
        random_state=42
    )

    best_xgb = XGBClassifier(
        n_estimators=200,
        learning_rate=0.05,
        max_depth=5,
        subsample=0.9,
        colsample_bytree=0.9,
        scale_pos_weight=scale_pos_weight_value,
        eval_metric="logloss",
        random_state=42
    )

    stacking_model =  StackingClassifier( 
            estimators=[
                ('rf', best_rf),
                ('xgb', best_xgb)
            ],
            final_estimator=LogisticRegression(max_iter=1000), 
            stack_method='predict_proba', # 베이스 모델의 확률을 최종 모델의 특징으로 사용
            cv=cv_folds,
            n_jobs=-1
        )

    stacking_model.fit(X_train, y_train)
    
    return stacking_model


def train_logistic_regression(
    X_train,
    y_train,
):
    """로지스틱 회귀 모델 학습"""
    model = LogisticRegression(max_iter=1000, random_state=42)
    model.fit(X_train, y_train)
    return model

def evaluate_model(
    model,
    X_test,
    y_test,
):
    """모델 예측 및 평가 결과 출력"""
    y_pred = model.predict(X_test)
    y_proba = model.predict_proba(X_test)[:, 1]
    acc = accuracy_score(y_test, y_pred)
    roc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    print(f"정확도: {acc:.4f}")
    print(f"ROC-AUC: {roc:.4f}")
    print(f"f1-score: {f1:.4f}")
    print(f"recall: {recall:.4f}")
    print(f"precision: {precision:.4f}")
    print("\n분류 리포트:")
    print(classification_report(y_test, y_pred))
