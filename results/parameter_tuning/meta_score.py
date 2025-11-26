import pandas as pd


# =========================
# F2-score 계산 함수
# F2 = (1 + 2^2) * (precision * recall) / (4*precision + recall)
# =========================
def compute_f2(precision, recall):
    """
    F2-score를 계산합니다.
    
    Args:
        precision: 정밀도 값
        recall: 재현율 값
    
    Returns:
        float: F2-score 값
    """
    if precision + recall == 0:
        return 0
    beta = 2
    return (1 + beta**2) * (precision * recall) / ((beta**2 * precision) + recall)


# =========================
# 점수 계산 함수
# =========================
def calculate_scores(df):
    """
    DataFrame에 f2와 meta_score를 계산하여 추가합니다.
    
    Args:
        df: 입력 DataFrame
    
    Returns:
        DataFrame: f2와 meta_score가 추가된 DataFrame
    """
    # F2 계산
    df["f2"] = df.apply(lambda x: compute_f2(x["precision"], x["recall"]), axis=1)
    
    # MetaScore 계산 (공식: 0.5*F2 + 0.3*PR-AUC + 0.2*ROC-AUC)
    df["meta_score"] = (
        0.5 * df["f2"] +
        0.3 * df["pr_auc"] +
        0.2 * df["roc_auc"]
    )
    
    return df


# =========================
# 결과 출력 함수
# =========================
def display_results(df):
    """
    MetaScore 기준으로 정렬된 결과를 출력합니다.
    
    Args:
        df: 점수가 계산된 DataFrame
    """
    df_sorted = df.sort_values("meta_score", ascending=False)
    
    print("\n=== Top Models by MetaScore ===")
    print(df_sorted[[
        "ensemble_strategy",
        "n_trials",
        "accuracy",
        "roc_auc",
        "pr_auc",
        "precision",
        "recall",
        "f1",
        "f2",
        "meta_score"
    ]].head(10))
    
    # Best model info
    best_row = df_sorted.iloc[0]
    print("\n=== Best Model Based on MetaScore ===")
    print(best_row)


# =========================
# 최고 모델의 n_trials 확인 함수
# =========================
def get_best_n_trials(df):
    """
    meta_score가 가장 높은 모델의 n_trials를 반환합니다.
    
    Args:
        df: 점수가 계산된 DataFrame (meta_score 컬럼 포함)
    
    Returns:
        int: 최고 meta_score를 가진 모델의 n_trials 값
    """
    if "meta_score" not in df.columns:
        raise ValueError("DataFrame에 'meta_score' 컬럼이 없습니다. 먼저 calculate_scores()를 실행하세요.")
    
    best_row = df.loc[df["meta_score"].idxmax()]
    return int(best_row["n_trials"])


def get_best_model_info(df):
    """
    meta_score가 가장 높은 모델의 전체 정보를 반환합니다.
    
    Args:
        df: 점수가 계산된 DataFrame (meta_score 컬럼 포함)
    
    Returns:
        Series: 최고 meta_score를 가진 모델의 전체 정보
    """
    if "meta_score" not in df.columns:
        raise ValueError("DataFrame에 'meta_score' 컬럼이 없습니다. 먼저 calculate_scores()를 실행하세요.")
    
    best_row = df.loc[df["meta_score"].idxmax()]
    return best_row


# =========================
# Ensemble별 최적 파라미터 추출 함수
# =========================
def get_best_parameters_by_ensemble(df, output_file="best_n_trials_parameter.csv"):
    """
    ensemble_strategy별로 meta_score가 가장 높은 모델의 모든 성능 지표와 n_trials를 추출하여 CSV로 저장합니다.
    
    Args:
        df: 점수가 계산된 DataFrame (meta_score 컬럼 포함)
        output_file: 출력 파일명 (기본값: "best_n_trials_parameter.csv")
    
    Returns:
        DataFrame: ensemble_strategy별 최적 모델 정보
    """
    if "meta_score" not in df.columns:
        raise ValueError("DataFrame에 'meta_score' 컬럼이 없습니다. 먼저 calculate_scores()를 실행하세요.")
    
    if "ensemble_strategy" not in df.columns:
        raise ValueError("DataFrame에 'ensemble_strategy' 컬럼이 없습니다.")
    
    # ensemble_strategy별로 그룹화하고 meta_score가 최대인 행 추출
    best_models = df.loc[df.groupby("ensemble_strategy")["meta_score"].idxmax()].copy()
    
    # 정렬 (ensemble_strategy 기준)
    best_models = best_models.sort_values("ensemble_strategy").reset_index(drop=True)
    
    # CSV 파일로 저장
    best_models.to_csv(output_file, index=False)
    
    print(f"\n=== Ensemble별 최적 파라미터 저장 완료 ===")
    print(f"파일명: {output_file}")
    print(f"총 {len(best_models)}개 ensemble strategy의 최적 모델이 저장되었습니다.")
    print("\n=== Ensemble별 최적 모델 정보 ===")
    print(best_models[["ensemble_strategy", "n_trials", "accuracy", "roc_auc", "pr_auc", 
                       "precision", "recall", "f1", "f2", "meta_score"]])
    
    return best_models


# =========================
# CSV 저장 함수
# =========================
def save_results(df, output_file="n_trials_comparison.csv"):
    """
    계산된 결과를 CSV 파일로 저장합니다.
    
    Args:
        df: 저장할 DataFrame
        output_file: 출력 파일명 (기본값: "n_trials_comparison.csv")
    """
    df.to_csv(output_file, index=False)
    print(f"\n=== 결과 저장 완료 ===")
    print(f"파일명: {output_file}")
    print(f"총 {len(df)}개 행이 저장되었습니다.")


# =========================
# 메인 함수
# =========================
def main(input_file="n_trials_comparison.csv", output_file=None):
    """
    메인 실행 함수
    
    Args:
        input_file: 입력 CSV 파일명 (기본값: "n_trials_comparison.csv")
        output_file: 출력 CSV 파일명 (None이면 input_file과 동일)
    """
    if output_file is None:
        output_file = input_file
    
    # 1. CSV 파일 로드
    print(f"=== CSV 파일 로드 중: {input_file} ===")
    df = pd.read_csv(input_file)
    print(f"총 {len(df)}개 행을 로드했습니다.")
    
    # 2. 점수 계산
    print("\n=== 점수 계산 중 (f2, meta_score) ===")
    df = calculate_scores(df)
    
    # 3. 결과 출력
    display_results(df)
    
    # 4. 최고 모델의 n_trials 확인
    best_n_trials = get_best_n_trials(df)
    print(f"\n=== 최고 MetaScore 모델의 n_trials ===")
    print(f"n_trials: {best_n_trials}")
    
    # 5. Ensemble별 최적 파라미터 추출 및 저장
    get_best_parameters_by_ensemble(df, "best_n_trials_parameter.csv")
    
    # 6. 결과 저장
    save_results(df, output_file)


if __name__ == "__main__":
    main()