from sklearn.model_selection import train_test_split

def split_train_test(df, target_col='Attrition_Binary', test_size=0.2, random_state=42):
	"""학습/테스트 데이터 분리"""
	X = df.drop(target_col, axis=1)
	y = df[target_col]
	X_train, X_test, y_train, y_test = train_test_split(
		X, y, test_size=test_size, random_state=random_state, stratify=y
	)
	return X_train, X_test, y_train, y_test
