"""Простые проверки src/data.py: фичи, кодирование и отсутствие NaN."""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from src.data import (CAT_COLS, SERVICE_COLS, TARGET, add_features,
                      build_datasets, encode, transform_for_inference)


def make_raw(n: int = 6, seed: int = 0) -> pd.DataFrame:
    """Маленький synthetic-датасет со всеми колонками, которые нужны add_features."""
    rng = np.random.default_rng(seed)
    df = pd.DataFrame({c: rng.integers(0, 6, n) for c in SERVICE_COLS})
    df["Age"] = rng.integers(18, 80, n)
    df["Flight Distance"] = rng.integers(300, 5000, n)
    df["Departure Delay in Minutes"] = rng.integers(0, 300, n)
    df["Arrival Delay in Minutes"] = rng.integers(0, 300, n)
    df["Gender"] = rng.choice(["Male", "Female"], n)
    df["Customer Type"] = rng.choice(["Loyal Customer", "disloyal Customer"], n)
    df["Type of Travel"] = rng.choice(["Business travel", "Personal travel"], n)
    df["Class"] = rng.choice(["Business", "Eco", "Eco Plus"], n)
    df["id"] = np.arange(n)  # колонка id есть и в test.csv
    return df


def test_add_features_adds_service_mean():
    """add_features добавляет service_mean = среднему по сервисным колонкам."""
    df = make_raw()
    out = add_features(df)

    assert "service_mean" in out.columns
    expected = df[SERVICE_COLS].mean(axis=1)
    np.testing.assert_allclose(out["service_mean"].to_numpy(), expected.to_numpy())
    # исходные колонки не потеряны
    assert "Inflight wifi service" in out.columns


def test_encode_gives_same_columns_for_train_and_test():
    """После encode у train и test одинаковые колонки (в одном порядке)."""
    train = make_raw()
    train[TARGET] = [0, 1, 0, 1, 0, 1]
    test = make_raw(seed=1)

    X, X_test = encode(train, test)

    assert list(X.columns) == list(X_test.columns)
    assert TARGET not in X.columns
    assert len(X) == len(train) and len(X_test) == len(test)
    # все категориальные колонки превратились в числовые (dummies)
    assert not any(c in X.columns for c in CAT_COLS)


def test_transform_for_inference_matches_encode():
    """transform_for_inference(test) даёт те же колонки (и порядок), что encode на train."""
    train = make_raw()
    train[TARGET] = [0, 1, 0, 1, 0, 1]
    test_raw = make_raw(seed=1)

    X, _ = encode(train, test_raw)
    ref_cols = [c for c in X.columns if c != "id"]  # как в build_datasets

    X_new, ids = transform_for_inference(test_raw, ref_cols)

    assert list(X_new.columns) == ref_cols
    assert len(X_new) == len(test_raw) and len(ids) == len(test_raw)
    assert not X_new.isna().any().any()
    assert "id" not in X_new.columns


def test_transform_for_inference_unknown_category():
    """Неизвестная категория («Space») не ломает инференс: её dummy-колонки = 0."""
    train = make_raw()
    train[TARGET] = [0, 1, 0, 1, 0, 1]
    test_raw = make_raw(seed=1)
    X, _ = encode(train, test_raw)
    ref_cols = [c for c in X.columns if c != "id"]

    # подменяем категорию на выдуманную — при обучении её не было
    test_raw.loc[0, "Class"] = "Space"

    X_new, _ = transform_for_inference(test_raw, ref_cols)

    assert list(X_new.columns) == ref_cols  # не упал и колонки сходятся
    class_dummies = [c for c in ref_cols if c.startswith("Class_")]
    assert (X_new.loc[0, class_dummies] == 0).all()  # строка с "Space" не попала ни в одну dummy


def test_build_datasets_no_nan_and_aligned_columns():
    """Реальные данные: нет NaN, train и test имеют одинаковые фичи."""
    root = Path(__file__).resolve().parents[1]
    paths = {
        "train": root / "data" / "raw" / "train.csv",
        "test": root / "data" / "raw" / "test.csv",
        "sample_submission": root / "data" / "raw" / "sample_submission.csv",
    }
    if not paths["train"].exists():
        pytest.skip("нет data/raw/train.csv — положи данные, чтобы пройти этот тест")
    X, y, X_test, test_ids, sub = build_datasets(paths)

    assert X.shape[1] == X_test.shape[1]
    assert list(X.columns) == list(X_test.columns)
    assert not X.isna().any().any(), "в X есть NaN"
    assert not X_test.isna().any().any(), "в X_test есть NaN"
    assert len(y) == len(X) and set(np.unique(y)) == {0, 1}
    assert len(test_ids) == len(X_test) == len(sub)
