from __future__ import annotations

from typing import Any, Callable, Dict, Tuple

import numpy as np
from sklearn.model_selection import GridSearchCV, RandomizedSearchCV, cross_val_score
from sklearn.preprocessing import StandardScaler
import optuna
from scipy.stats import loguniform
from .cv import stratified_kfold_split

def _fit_estimator(estimator: Any, X: np.ndarray, y: np.ndarray) -> Any:
    """Fit a sklearn‑compatible estimator and return the fitted model.

    Parameters
    ----------
    estimator : Any
        An unfitted estimator (e.g., LogisticRegression()).
    X, y : np.ndarray
        Training data.
    """
    estimator.fit(X, y)
    return estimator


def grid_search_tuner(
    estimator: Any,
    param_grid: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: Any = 5,
    groups: np.ndarray | None = None,
    scoring: str | None = None,
    n_jobs: int = -1,
) -> Tuple[Any, Dict[str, Any], float]:
    """Exhaustive grid search.

    Returns the best estimator, its hyper‑parameters and the best CV score.
    """
    # cv값이 cv 분리하는 함수이면 그 방식으로 fold 생성, int이면 cv만큼 fold 생성
    if callable(cv) and not isinstance(cv, int):
        cv_obj = cv(groups) if groups is not None else cv()
    else:
        cv_obj = cv

    grid = GridSearchCV(
        estimator,
        param_grid,
        cv=cv_obj,
        scoring=scoring,
        n_jobs=n_jobs,
        refit=True,
    )
    grid.fit(X, y)
    return grid.best_estimator_, grid.best_params_, grid.best_score_


def random_search_tuner(
    estimator: Any,
    param_distributions: Dict[str, Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: Any = 5,
    groups: np.ndarray | None = None,
    scoring: str | None = None,
    n_iter: int = 50,
    n_jobs: int = -1,
    random_state: int = 42,
) -> Tuple[Any, Dict[str, Any], float]:
    """Randomized search over hyper‑parameter distributions.

    ``n_iter`` controls how many random configurations are tried.
    """
    # cv값이 cv 분리하는 함수이면 그 방식으로 fold 생성, int이면 cv만큼 fold 생성
    if callable(cv) and not isinstance(cv, int):
        cv_obj = cv(groups) if groups is not None else cv()
    else:
        cv_obj = cv

    rand = RandomizedSearchCV(
        estimator,
        param_distributions,
        n_iter=n_iter,
        cv=cv_obj,
        scoring=scoring,
        n_jobs=n_jobs,
        random_state=random_state,
        refit=True,
    )
    rand.fit(X, y)
    return rand.best_estimator_, rand.best_params_, rand.best_score_


def optuna_tuner(
    estimator_factory: Callable[[optuna.Trial], Any],
    X: np.ndarray,
    y: np.ndarray,
    cv: Any = 5,
    groups: np.ndarray | None = None,
    scoring: str = "roc_auc",
    n_trials: int = 50,
    timeout: int | None = None,
    direction: str = "maximize",
) -> Tuple[Any, Dict[str, Any], float]:
    """Hyper‑parameter optimisation with Optuna.

    Returns the best estimator (trained on the full data), the best params and the best CV score.
    """

    # cv값이 cv 분리하는 함수이면 그 방식으로 fold 생성, int이면 cv만큼 fold 생성
    if callable(cv) and not isinstance(cv, int):
        cv_obj = cv(groups) if groups is not None else cv()
    else:
        cv_obj = cv
    def objective(trial: optuna.Trial) -> float:
        estimator = estimator_factory(trial)
        scores = cross_val_score(
            estimator, X, y, cv=cv_obj, scoring=scoring, n_jobs=-1
        )
        return np.mean(scores)


    study = optuna.create_study(direction=direction)
    study.optimize(objective, n_trials=n_trials, timeout=timeout)

    best_params = study.best_trial.params
    best_estimator = estimator_factory(optuna.trial.FixedTrial(best_params))
    best_estimator = _fit_estimator(best_estimator, X, y)
    best_score = study.best_value
    return best_estimator, best_params, best_score


# ---------------------------------------------------------------------------
# Example usage (remove or comment out in production code)
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    from sklearn.datasets import load_breast_cancer
    from sklearn.linear_model import LogisticRegression

    # Load a toy dataset
    data = load_breast_cancer()
    X, y = data.data, data.target

    # Scaling often helps convergence for logistic regression
    scaler = StandardScaler()
    X = scaler.fit_transform(X)

    # Grid search – increase max_iter for reliable convergence
    param_grid = {"C": [0.01, 0.1, 1, 10], "penalty": ["l2"]}
    best_est, best_cfg, best_sc = grid_search_tuner(
        LogisticRegression(max_iter=1000, solver="lbfgs"), param_grid, X, y
    )
    print("GridSearch best score:", best_sc, "params:", best_cfg)

    # 2️⃣ Random search – use scipy.stats.loguniform for a continuous log‑uniform distribution
    param_dist = {"C": loguniform(1e-3, 1e2), "penalty": ["l2"]}
    best_est, best_cfg, best_sc = random_search_tuner(
        LogisticRegression(max_iter=1000, solver="lbfgs"), param_dist, X, y, n_iter=30
    )
    print("RandomSearch best score:", best_sc, "params:", best_cfg)

    # 3️⃣ Optuna – use suggest_float(log=True) instead of the deprecated suggest_loguniform
    def make_lr(trial: optuna.Trial) -> LogisticRegression:
        C = trial.suggest_float("C", 1e-3, 1e2, log=True)
        return LogisticRegression(C=C, max_iter=1000, solver="lbfgs")

    best_est, best_cfg, best_sc = optuna_tuner(make_lr, X, y, n_trials=30)
    print("Optuna best score:", best_sc, "params:", best_cfg)