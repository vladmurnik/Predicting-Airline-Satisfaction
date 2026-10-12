# airline-satisfaction

Предсказание удовлетворённости пассажиров авиакомпании — Kaggle
[Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10),
метрика — **ROC-AUC**.

Пайплайн: загрузка данных → чистка → простые фичи → one-hot → XGBoost + LightGBM
(5-fold CV, early stopping) → среднее (blend) → submission.

## Быстрый старт

```bash
# 1) зависимости (создай venv и активируй его, если ещё нет)
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2) положи данные в data/raw/: train.csv, test.csv, sample_submission.csv

# обучение: печатает OOF AUC по фолдам и пишет submissions/submission.csv
python train.py

# быстрый smoke-тест (в отдельную папку, чтобы не затереть настоящий submission)
python train.py --model xgb --sample-train 20000 --n-estimators 200 --submissions-dir submissions/smoke

# тесты data-слоя
python -m pytest
```

В этом репозитории уже есть Windows-окружение `.venv/` (в git не входит) — вместо
`python` можно писать `.venv/Scripts/python.exe`. Флаги: `--model {xgb,lgb,blend}`
(по умолчанию `blend`), `--folds`, `--seed`, `--n-estimators`, `--sample-train N`,
`--submissions-dir`, `--outdir` (подробнее — `train.py --help`).

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
├── pyproject.toml      # конфиг pytest
├── requirements.txt
└── README.md
```

## Результат (OOF AUC, 5-fold, замерено на полных данных)

| Вариант | OOF AUC |
|---------|---------|
| XGBoost | 0.9587 |
| LightGBM | 0.9587 |
| **Blend (XGB + LGB, среднее)** | **0.9588** |

Research-ансамбль (blend_05 — 146 OOF, перебор seed'ов и весов) доходил до **0.96186
на лидерборде** (OOF 0.9622), но он убран ради простоты: ради +0.003 AUC — сотни
файлов и часы расчётов. Старое лежит в `research_archive/` и в работе не участвует.

## Формат submission

`submissions/submission.csv`:

- ровно две колонки: `id,satisfaction`;
- `satisfaction` — вероятность в `[0,1]` (`predict_proba(...)[:, 1]`), не 0/1 и не строки;
- 299 844 строки, порядок строк как в `sample_submission.csv`.
