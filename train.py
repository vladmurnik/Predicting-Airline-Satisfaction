#!/usr/bin/env python
"""Точка входа: данные -> 5-fold CV -> OOF AUC -> submissions/submission.csv."""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import build_datasets
from src.model import cv_predict

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="CV-обучение XGBoost/LightGBM и запись submission")
    p.add_argument("--folds", type=int, default=5, help="число фолдов (по умолчанию 5)")
    p.add_argument("--seed", type=int, default=42, help="random_state разбиения и моделей")
    p.add_argument("--model", choices=["xgb", "lgb", "blend"], default="blend",
                   help="blend = среднее предсказаний XGB и LGB (по умолчанию)")
    p.add_argument("--sample-train", type=int, default=None, metavar="N",
                   help="обучаться на N случайных строках train (быстрый smoke-тест)")
    p.add_argument("--n-estimators", type=int, default=None,
                   help="переопределить число деревьев (удобно для smoke-теста)")
    p.add_argument("--outdir", type=Path, default=ROOT / "outputs",
                   help="куда сохранять oof_*.npy / pred_*.npy")
    p.add_argument("--submissions-dir", type=Path, default=ROOT / "submissions",
                   help="куда сохранять submission.csv")
    return p.parse_args()


def subsample(X, y, n, seed):
    """Берёт n случайных строк train, если задан --sample-train."""
    if n is None or n >= len(X):
        return X, y
    idx = np.random.default_rng(seed).permutation(len(X))[:n]
    return X.iloc[idx], y[idx]


def save_submission(pred, test_ids, path):
    """Проверяет, что pred — вероятности, и пишет id,satisfaction."""
    assert pred.min() >= 0.0 and pred.max() <= 1.0, "satisfaction должна быть вероятностью в [0,1]"
    pd.DataFrame({"id": test_ids, "satisfaction": pred}).to_csv(path, index=False)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # кириллица в выводе не ломается на Windows
    a = parse_args()

    paths = {
        "train": ROOT / "data" / "raw" / "train.csv",
        "test": ROOT / "data" / "raw" / "test.csv",
        "sample_submission": ROOT / "data" / "raw" / "sample_submission.csv",
    }
    X, y, X_test, test_ids, _ = build_datasets(paths)
    X, y = subsample(X, y, a.sample_train, a.seed)
    print(f"train {X.shape} | test {X_test.shape} | model={a.model} folds={a.folds} seed={a.seed}")

    # Обучаем нужные модели, у каждой запоминаем OOF и прогноз на test
    runs = {}
    for name in (["xgb", "lgb"] if a.model == "blend" else [a.model]):
        print(f"\n== {name} ==")
        oof, pred, auc = cv_predict(X, y, X_test, n_splits=a.folds, seed=a.seed,
                                    model_name=name, n_estimators=a.n_estimators)
        runs[name] = (oof, pred, auc)
        print(f"OOF AUC ({name}) = {auc:.5f}")

    if a.model == "blend":
        oof = 0.5 * runs["xgb"][0] + 0.5 * runs["lgb"][0]
        pred = 0.5 * runs["xgb"][1] + 0.5 * runs["lgb"][1]
        auc = roc_auc_score(y, oof)
        print(f"OOF AUC (blend) = {auc:.5f}")
    else:
        oof, pred, auc = runs[a.model]

    a.outdir.mkdir(parents=True, exist_ok=True)
    np.save(a.outdir / f"oof_{a.model}.npy", oof)
    np.save(a.outdir / f"pred_{a.model}.npy", pred)

    a.submissions_dir.mkdir(parents=True, exist_ok=True)
    sub_path = a.submissions_dir / "submission.csv"
    save_submission(pred, test_ids, sub_path)

    print(f"\nГотово: OOF AUC = {auc:.5f} | submission -> {sub_path}")


if __name__ == "__main__":
    main()
