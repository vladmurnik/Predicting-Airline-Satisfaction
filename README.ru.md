[English](README.md) | **Русский**

# Предсказание удовлетворённости авиапассажиров

Соревнование Kaggle [Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10),
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

# 2) качай данные со страницы соревнования Kaggle (Playground Series S6E10) →
#    вкладка Data → train.csv, test.csv, sample_submission.csv → в data/raw/

# обучение: печатает OOF AUC по фолдам и пишет submissions/submission.csv
python train.py

# быстрый smoke-тест (в отдельную папку, чтобы не затереть настоящий submission)
python train.py --model xgb --sample-train 20000 --n-estimators 200 --submissions-dir submissions/smoke

# тесты data-слоя
python -m pytest
```

В командах ниже `python` — интерпретатор твоего окружения
(`.venv/Scripts/python.exe` на Windows). Флаги: `--model {xgb,lgb,blend}`
(по умолчанию `blend`), `--folds`, `--seed`, `--n-estimators`, `--sample-train N`,
`--submissions-dir`, `--outdir`, `--save-model`, `--model-dir` (подробнее — `train.py --help`).

## Инференс

Чтобы предсказывать сохранёнными моделями (без переобучения):

```bash
# 1) обучить и сохранить финальные модели в outputs/model/
python train.py --save-model

# 2) предсказать из любого csv с теми же колонками и записать submission
python predict.py
```

`predict.py --input путь/к/файлу.csv` принимает любой csv с теми же колонками,
что и `data/raw/test.csv` (колонка `satisfaction` не нужна). По умолчанию читает
`data/raw/test.csv` и пишет `submissions/submission.csv`.

## Структура

```
.
├── train.py            # точка входа: CV + OOF AUC + submission (+ --save-model)
├── predict.py          # инференс сохранёнными моделями из outputs/model/
├── src/
│   ├── data.py         # загрузка, чистка, фичи, encode, инференс-трансформ, фолды
│   └── model.py        # XGBoost/LightGBM, cv_predict, fit_final (early stopping)
├── tests/
│   └── test_data.py    # тесты data-слоя
├── data/raw/           # train.csv / test.csv / sample_submission.csv (не в git)
├── outputs/model/      # финальные модели *.joblib + feature_columns.json (не в git)
├── pyproject.toml      # конфиг pytest
├── requirements.txt
├── LICENSE             # MIT-лицензия
├── .gitignore          # что не попадает в git (данные, outputs, .venv…)
├── README.md           # english
└── README.ru.md        # этот файл
```

## Результат (OOF AUC, 5-fold, замерено на полных данных)

| Вариант | OOF AUC |
|---------|---------|
| XGBoost | 0.9587 |
| LightGBM | 0.9587 |
| **Blend (XGB + LGB, среднее)** | **0.9588** |

Пример вывода `python train.py` (полный прогон; блок сокращён — опущены
промежуточные fold-строки, путь к submission показан относительным,
в настоящем логе он абсолютный):

```
train (699635, 39) | test (299844, 39) | model=blend folds=5 seed=42

== xgb ==
  fold 1/5: AUC = 0.95882
  ...
  fold 5/5: AUC = 0.95899
OOF AUC (xgb) = 0.95868

== lgb ==
  ...
OOF AUC (lgb) = 0.95871
OOF AUC (blend) = 0.95882

Готово: OOF AUC = 0.95882 | submission -> submissions/submission.csv
```

Research-ансамбль (blend_05 — 146 OOF, перебор seed'ов и весов) доходил до **0.96186
на лидерборде** (OOF 0.9622), но он убран ради простоты: ради +0.003 AUC — сотни
файлов и часы расчётов. Старое осталось только локально в `research_archive/`
(в git не входит) и в работе не участвует.

## Формат submission

`submissions/submission.csv`:

- ровно две колонки: `id,satisfaction`;
- `satisfaction` — вероятность в `[0,1]` (`predict_proba(...)[:, 1]`), не 0/1 и не строки;
- 299 844 строки, порядок строк как в `sample_submission.csv`.
