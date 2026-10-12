"""Модели: XGBoost и LightGBM, CV-предсказания с early stopping."""
import numpy as np
from lightgbm import LGBMClassifier, early_stopping as lgb_early_stopping
from sklearn.metrics import roc_auc_score
from xgboost import XGBClassifier

from src.data import cv_folds

EARLY_STOPPING_ROUNDS = 100

# Гиперпараметры зафиксированы. n_estimators намеренно большой:
# ранняя остановка оборвёт обучение, как только AUC на val-фолде не растёт.
XGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "max_depth": 8,
    "subsample": 0.85,
    "colsample_bytree": 0.75,
    "min_child_weight": 20,
    "reg_lambda": 3.0,
    "tree_method": "hist",
    "eval_metric": "logloss",
    "early_stopping_rounds": EARLY_STOPPING_ROUNDS,
}

LGB_PARAMS = {
    "n_estimators": 3000,
    "learning_rate": 0.02,
    "num_leaves": 64,
    "min_child_samples": 60,
    "subsample": 0.8,
    "subsample_freq": 1,
    "colsample_bytree": 0.75,
    "reg_lambda": 3.0,
    "verbose": -1,
}


def make_model(model_name: str, seed: int, n_estimators=None):
    """Новая модель с гиперпараметрами из XGB_PARAMS / LGB_PARAMS."""
    if model_name == "xgb":
        model_cls, params = XGBClassifier, XGB_PARAMS
    elif model_name == "lgb":
        model_cls, params = LGBMClassifier, LGB_PARAMS
    else:
        raise ValueError(f"model_name должен быть 'xgb' или 'lgb', получено {model_name!r}")

    params = {**params, "random_state": seed}
    if n_estimators is not None:
        params["n_estimators"] = n_estimators
    return model_cls(**params)


def fit_fold(model_name, X_tr, y_tr, X_va, y_va, seed, n_estimators=None):
    """Фитит модель на train-части фолда, early stopping на val-части."""
    model = make_model(model_name, seed, n_estimators)
    if model_name == "xgb":
        model.fit(X_tr, y_tr, eval_set=[(X_va, y_va)], verbose=False)
    else:
        model.fit(X_tr, y_tr, eval_X=X_va, eval_y=y_va,
                  callbacks=[lgb_early_stopping(EARLY_STOPPING_ROUNDS, verbose=False)])
    return model


def cv_predict(X, y, X_test, n_splits=5, seed=42, model_name="xgb", n_estimators=None):
    """Кросс-валидация: возвращает (oof, средний прогноз на test, OOF AUC).

    OOF — вероятность для каждой строки train, предсказанная моделью,
    которая на ней не обучалась. Прогноз на test — среднее по фолдам.
    """
    oof = np.zeros(len(X))
    test_preds = np.zeros(len(X_test))

    for fold, (tr, va) in enumerate(cv_folds(y, n_splits, seed), start=1):
        X_tr, X_va = X.iloc[tr], X.iloc[va]
        model = fit_fold(model_name, X_tr, y[tr], X_va, y[va],
                         seed=seed + fold, n_estimators=n_estimators)  # разный seed на каждый фолд
        oof[va] = model.predict_proba(X_va)[:, 1]
        test_preds += model.predict_proba(X_test)[:, 1] / n_splits
        print(f"  fold {fold}/{n_splits}: AUC = {roc_auc_score(y[va], oof[va]):.5f}")

    return oof, test_preds, roc_auc_score(y, oof)
