#!/usr/bin/env python
"""Точка входа: данные -> 5-fold CV -> OOF AUC -> submissions/submission.csv."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

from src.data import build_datasets
from src.model import cv_predict, fit_final

ROOT = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="CV-обучение XGBoost/LightGBM и запись submission")
    parser.add_argument("--folds", type=int, default=5, help="число фолдов (по умолчанию 5)")
    parser.add_argument("--seed", type=int, default=42, help="random_state разбиения и моделей")
    parser.add_argument("--model", choices=["xgb", "lgb", "blend"], default="blend",
                        help="blend = среднее предсказаний XGB и LGB (по умолчанию)")
    parser.add_argument("--sample-train", type=int, default=None, metavar="N",
                        help="обучаться на N случайных строках train (быстрый smoke-тест)")
    parser.add_argument("--n-estimators", type=int, default=None,
                        help="переопределить число деревьев (удобно для smoke-теста)")
    parser.add_argument("--outdir", type=Path, default=ROOT / "outputs",
                        help="куда сохранять oof_*.npy / pred_*.npy")
    parser.add_argument("--submissions-dir", type=Path, default=ROOT / "submissions",
                        help="куда сохранять submission.csv")
    parser.add_argument("--save-model", action="store_true",
                        help="дообучить финальные модели на всём train и сохранить в --model-dir; "
                             "при --sample-train модель обучится на подвыборке — только для проверки")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "outputs" / "model",
                        help="куда сохранять финальные модели (*.joblib) и список колонок")
    return parser.parse_args()


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
    args = parse_args()

    paths = {
        "train": ROOT / "data" / "raw" / "train.csv",
        "test": ROOT / "data" / "raw" / "test.csv",
        "sample_submission": ROOT / "data" / "raw" / "sample_submission.csv",
    }
    X, y, X_test, test_ids, _ = build_datasets(paths)
    X, y = subsample(X, y, args.sample_train, args.seed)
    print(f"train {X.shape} | test {X_test.shape} | model={args.model} folds={args.folds} seed={args.seed}")

    # учим нужные модели и запоминаем их результаты
    results = {}
    for name in (["xgb", "lgb"] if args.model == "blend" else [args.model]):
        print(f"\n== {name} ==")
        oof, pred, auc, iters = cv_predict(X, y, X_test, n_splits=args.folds, seed=args.seed,
                                           model_name=name, n_estimators=args.n_estimators)
        results[name] = {"oof": oof, "pred": pred, "auc": auc, "iters": iters}
        print(f"OOF AUC ({name}) = {auc:.5f}")

    if args.model == "blend":
        xgb, lgb = results["xgb"], results["lgb"]
        oof = 0.5 * xgb["oof"] + 0.5 * lgb["oof"]
        pred = 0.5 * xgb["pred"] + 0.5 * lgb["pred"]
        auc = roc_auc_score(y, oof)
        print(f"OOF AUC (blend) = {auc:.5f}")
    else:
        one = results[args.model]
        oof, pred, auc = one["oof"], one["pred"], one["auc"]

    args.outdir.mkdir(parents=True, exist_ok=True)
    np.save(args.outdir / f"oof_{args.model}.npy", oof)
    np.save(args.outdir / f"pred_{args.model}.npy", pred)

    args.submissions_dir.mkdir(parents=True, exist_ok=True)
    sub_path = args.submissions_dir / "submission.csv"
    save_submission(pred, test_ids, sub_path)

    # финальные модели на всём train — их загружает predict.py
    if args.save_model:
        args.model_dir.mkdir(parents=True, exist_ok=True)
        print(f"\nСохраняю финальные модели в {args.model_dir}")
        for name, res in results.items():
            # число деревьев = медиана итераций по фолдам ЭТОЙ модели
            n_final = int(np.median(res["iters"]))
            print(f"  {name}: дообучаю на {n_final} деревьях (итерации по фолдам: {res['iters']})")
            model = fit_final(name, X, y, args.seed, n_final)
            joblib.dump(model, args.model_dir / f"model_{name}.joblib")
        # список колонок, по которым обучались модели — predict.py строит вход по нему
        with open(args.model_dir / "feature_columns.json", "w") as f:
            json.dump({"columns": list(X.columns)}, f)
        if args.sample_train:
            print("ВНИМАНИЕ: модель обучена на подвыборке (--sample-train), для боевого инференса не использовать")

    print(f"\nГотово: OOF AUC = {auc:.5f} | submission -> {sub_path}")


if __name__ == "__main__":
    main()
