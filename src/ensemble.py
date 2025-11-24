from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score, classification_report

def train_logistic_regression(
	X_train, 
	y_train
	):
	"""로지스틱 회귀 모델 학습"""
	model = LogisticRegression(max_iter=1000, random_state=42)
	model.fit(X_train, y_train)
	return model

def evaluate_model(
	model, 
	X_test, 
	y_test
	):
	"""모델 예측 및 평가 결과 출력"""
	y_pred = model.predict(X_test)
	y_proba = model.predict_proba(X_test)[:, 1]
	acc = accuracy_score(y_test, y_pred)
	roc = roc_auc_score(y_test, y_proba)
	print(f"정확도: {acc:.4f}")
	print(f"ROC-AUC: {roc:.4f}")
	print("\n분류 리포트:")
	print(classification_report(y_test, y_pred))
