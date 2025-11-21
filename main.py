from src.cv import split_train_test, train_test_split
from src.ensemble import train_logistic_regression, evaluate_model
from src.preprocessing import load_data, preprocess_data   

# 데이터 불러오기 및 전처리

if __name__ == '__main__':
    df = load_data()
    df = preprocess_data(df)

    X_train, X_test, y_train, y_test = split_train_test(
        df, target_col='Attrition_Binary',
        test_size=0.2, 
        random_state=42
        )
    
    model = train_logistic_regression(X_train, y_train)
    evaluate_model(model, X_test, y_test)