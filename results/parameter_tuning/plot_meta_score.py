"""n_trials에 따른 meta_score 변화를 시각화합니다."""

from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


DEFAULT_INPUT = Path(__file__).parent.parent.parent / "data" / "processed" / "parameter_tuning" / "n_trials_comparison.csv"
DEFAULT_OUTPUT = Path(__file__).parent / "figures" / "meta_score_by_n_trials.png"


def load_data(csv_path: Path) -> pd.DataFrame:
    """CSV 데이터를 로드하고 n_trials 기준으로 정렬합니다."""
    df = pd.read_csv(csv_path)
    if "n_trials" not in df.columns or "meta_score" not in df.columns:
        raise ValueError("필수 컬럼(n_trials, meta_score)이 없습니다.")
    return df.sort_values("n_trials")


def plot_meta_score_trend(df: pd.DataFrame, output_path: Path) -> None:
    """ensemble_strategy 별 meta_score 꺾은선 그래프를 저장합니다."""
    plt.style.use("seaborn-v0_8-whitegrid")
    fig, ax = plt.subplots(figsize=(11, 6))

    palette = {
        "voting": "#0969da",  # GitHub 블루
        "stacking": "#8250df",  # GitHub 퍼플
    }

    for strategy, group in df.groupby("ensemble_strategy"):
        color = palette.get(strategy, None)
        ax.plot(
            group["n_trials"],
            group["meta_score"],
            marker="o",
            linewidth=2.5,
            markersize=6,
            label=strategy.capitalize(),
            color=color,
        )

    ax.set_title("Meta Score by n_trials", fontsize=16, fontweight="bold")
    ax.set_xlabel("n_trials", fontsize=12)
    ax.set_ylabel("meta_score", fontsize=12)
    ax.legend(title="Ensemble Strategy")
    ax.grid(True, linestyle="--", linewidth=0.6, alpha=0.6)
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=300)
    plt.close(fig)


def main(input_path: Path = DEFAULT_INPUT, output_path: Path = DEFAULT_OUTPUT) -> None:
    df = load_data(input_path)
    plot_meta_score_trend(df, output_path)
    print(f"그래프가 '{output_path}'에 저장되었습니다.")


if __name__ == "__main__":
    main()

