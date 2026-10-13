#!/usr/bin/env python
"""Инференс: загружает сохранённые модели и пишет submission из любого csv с теми же колонками."""
import argparse
import json
import sys
from pathlib import Path

import joblib
import pandas as pd

from src.data import transform_for_inference

ROOT = Path(__file__).resolve().parent


def die(msg: str) -> None:
    """Печатает ошибку и выходит (без traceback)."""
    print(f"ОШИБКА: {msg}")
    sys.exit(1)


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")  # кириллица в выводе не ломается на Windows
    parser = argparse.ArgumentParser(description="Предсказание сохранёнными моделями (после train.py --save-model)")
    parser.add_argument("--model-dir", type=Path, default=ROOT / "outputs" / "model",
                        help="папка с model_*.joblib и feature_columns.json")
    parser.add_argument("--input", type=Path, default=ROOT / "data" / "raw" / "test.csv",
                        help="csv с теми же колонками, что и train (без target можно)")
    parser.add_argument("--output", type=Path, default=ROOT / "submissions" / "submission.csv",
                        help="куда записать submission.csv")
    args = parser.parse_args()

    # проверяем, что модели вообще сохранялись
    if not args.model_dir.is_dir():
        die(f"нет папки {args.model_dir} — сначала запусти python train.py --save-model")
    cols_path = args.model_dir / "feature_columns.json"
    if not cols_path.exists():
        die(f"нет {cols_path} — сначала запусти python train.py --save-model")
    model_paths = sorted(args.model_dir.glob("model_*.joblib"))
    if not model_paths:
        die(f"нет model_*.joblib в {args.model_dir} — сначала запусти python train.py --save-model")
    if not args.input.exists():
        die(f"нет файла {args.input} — укажи путь к csv через --input")

    with open(cols_path) as f:
        feature_columns = json.load(f)["columns"]
    X, ids = transform_for_inference(pd.read_csv(args.input), feature_columns)

    preds = [joblib.load(path).predict_proba(X)[:, 1] for path in model_paths]
    pred = sum(preds) / len(preds)  # среднее по моделям (blend)
    assert pred.min() >= 0.0 and pred.max() <= 1.0, "pred вне [0,1] — что-то пошло не так"

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"id": ids, "satisfaction": pred}).to_csv(args.output, index=False)
    print(f"Готово: {len(model_paths)} модель(и), {len(pred)} строк, pred в [{pred.min():.4f}, {pred.max():.4f}]")
    print(f"submission -> {args.output}")


if __name__ == "__main__":
    main()
