# airline-satisfaction

Предсказание удовлетворённости пассажиров авиакомпании — Kaggle
[Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10),
метрика — **ROC-AUC**.

Пайплайн: загрузка данных → чистка → простые фичи → one-hot → XGBoost + LightGBM
(5-fold CV, early stopping) → среднее (blend) → submission.

## Быстрый старт

```bash
# зависимости (venv уже есть в проекте)
.venv/Scripts/python.exe -m pip install -r requirements.txt

# обучение: печатает OOF AUC по фолдам и пишет submissions/submission.csv
.venv/Scripts/python.exe train.py

# быстрый smoke-тест вместо полного обучения
.venv/Scripts/python.exe train.py --model xgb --sample-train 20000 --n-estimators 200

# тесты data-слоя
.venv/Scripts/python.exe -m pytest
```

Нужны файлы `data/raw/train.csv`, `data/raw/test.csv`, `data/raw/sample_submission.csv`.
Флаги: `--model {xgb,lgb,blend}` (по умолчанию `blend`), `--folds`, `--seed`,
`--n-estimators`, `--sample-train N` (подробнее — `train.py --help`).

## Структура

```
.
├── train.py            # точка входа: CV + OOF AUC + submission
├── src/
│   ├── data.py         # загрузка, чистка, фичи, encode, фолды
│   └── model.py        # XGBoost/LightGBM, cv_predict (early stopping)
├── tests/
│   └── test_data.py    # тесты data-слоя
├── data/raw/           # train.csv / test.csv / sample_submission.csv (не в git)
├── requirements.txt
└── README.md
```

## Результат (OOF AUC, 5-fold)

| Вариант | OOF AUC |
|---------|---------|
| XGBoost | ~0.956 |
| LightGBM | ~0.956 |
| **Blend (XGB + LGB, среднее)** | **~0.957** |

Research-ансамбль (blend_05 — 146 OOF, перебор seed'ов и весов) доходил до **0.96186**,
но он убран ради простоты: ради +0.004 AUC — сотни файлов и часы расчётов.
Старое лежит в `research_archive/` и в работе не участвует.

## Формат submission

`submissions/submission.csv`:

- ровно две колонки: `id,satisfaction`;
- `satisfaction` — вероятность в `[0,1]` (`predict_proba(...)[:, 1]`), не 0/1 и не строки;
- 299 844 строки, порядок строк как в `sample_submission.csv`.
