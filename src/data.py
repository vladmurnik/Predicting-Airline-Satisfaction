"""Данные: загрузка, чистка, фичи, encode, инференс и стратифицированные фолды."""
import numpy as np
import pandas as pd
from sklearn.model_selection import StratifiedKFold

CAT_COLS = ["Gender", "Customer Type", "Type of Travel", "Class"]
TARGET = "satisfaction"
ID = "id"

SERVICE_COLS = [
    "Inflight wifi service",
    "Departure/Arrival time convenient",
    "Ease of Online booking",
    "Gate location",
    "Food and drink",
    "Online boarding",
    "Seat comfort",
    "Inflight entertainment",
    "On-board service",
    "Leg room service",
    "Baggage handling",
    "Checkin service",
    "Cleanliness",
]


def clean(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["Arrival Delay in Minutes"] = df["Arrival Delay in Minutes"].fillna(0.0)
    for c in CAT_COLS:
        df[c] = df[c].astype(str).str.strip()
    return df


def add_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["service_sum"] = df[SERVICE_COLS].sum(axis=1)
    df["service_mean"] = df[SERVICE_COLS].mean(axis=1)
    df["service_std"] = df[SERVICE_COLS].std(axis=1)
    df["service_min"] = df[SERVICE_COLS].min(axis=1)
    df["service_max"] = df[SERVICE_COLS].max(axis=1)
    df["zero_service_count"] = (df[SERVICE_COLS] == 0).sum(axis=1)
    df["total_delay"] = df["Departure Delay in Minutes"] + df["Arrival Delay in Minutes"]
    df["delay_diff"] = df["Departure Delay in Minutes"] - df["Arrival Delay in Minutes"]
    df["had_departure_delay"] = (df["Departure Delay in Minutes"] > 0).astype(int)
    df["dist_per_age"] = df["Flight Distance"] / (df["Age"] + 1)
    df["is_business"] = (df["Class"] == "Business").astype(int)
    df["is_loyal"] = (df["Customer Type"] == "Loyal Customer").astype(int)
    df["is_business_travel"] = (df["Type of Travel"] == "Business travel").astype(int)
    return df


def _dummies(df: pd.DataFrame) -> pd.DataFrame:
    """One-hot категориальных колонок — все get_dummies собраны в одном месте."""
    return pd.get_dummies(df, columns=CAT_COLS, drop_first=False)


def encode(train: pd.DataFrame, test: pd.DataFrame):
    # Склейка train+test перед get_dummies даёт одинаковые колонки у обоих
    full = pd.concat([train.drop(columns=[TARGET]), test], axis=0, ignore_index=True)
    full = _dummies(full)
    n = len(train)
    X = full.iloc[:n].reset_index(drop=True)
    X_test = full.iloc[n:].reset_index(drop=True)
    return X, X_test


def transform_for_inference(raw_df: pd.DataFrame, feature_columns: list):
    """Готовит новый csv к предсказанию моделью, обученной на train.

    Возвращает (X_new, ids): колонки строго как при обучении, в том же порядке.
    """
    ids = raw_df[ID]
    df = add_features(clean(raw_df))
    df = df.drop(columns=[TARGET], errors="ignore")  # если в csv есть target — убираем
    df = _dummies(df)
    df = df.drop(columns=[ID], errors="ignore")
    # ЯДРО надёжности: reindex выравнивает колонки под обучение —
    # лишние отбрасываются, недостающие заполняются нулями
    df = df.reindex(columns=feature_columns, fill_value=0)
    return df.astype(np.float32), ids


def build_datasets(paths: dict):
    train = pd.read_csv(paths["train"])
    test = pd.read_csv(paths["test"])
    sub = pd.read_csv(paths["sample_submission"])

    train = add_features(clean(train))
    test = add_features(clean(test))
    y = train[TARGET].astype(int).values
    test_ids = test[ID].values

    X, X_test = encode(train, test)
    X = X.drop(columns=[ID], errors="ignore").astype(np.float32)
    X_test = X_test.drop(columns=[ID], errors="ignore").astype(np.float32)
    return X, y, X_test, test_ids, sub


def cv_folds(y, n_splits=5, seed=42):
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=seed)
    return list(skf.split(np.zeros(len(y)), y))
