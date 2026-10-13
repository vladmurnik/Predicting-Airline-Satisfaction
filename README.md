**English** | [Русский](README.ru.md)

# Predicting Airline Passenger Satisfaction

A Kaggle competition:
[Playground Series S6E10](https://www.kaggle.com/competitions/playground-series-s6e10),
metric — **ROC-AUC**.

Pipeline: load data → clean → simple features → one-hot → XGBoost + LightGBM
(5-fold CV, early stopping) → average (blend) → submission.

## Quick start

```bash
# 1) dependencies (create and activate a venv if you don't have one)
python -m venv .venv
# Windows:      .venv\Scripts\activate
# Linux / macOS: source .venv/bin/activate
pip install -r requirements.txt

# 2) download the data from the Kaggle competition page (Playground Series S6E10) →
#    the Data tab → train.csv, test.csv, sample_submission.csv → into data/raw/

# train: prints per-fold OOF AUC and writes submissions/submission.csv
python train.py

# quick smoke test (into a separate folder so you don't overwrite the real submission)
python train.py --model xgb --sample-train 20000 --n-estimators 200 --submissions-dir submissions/smoke

# data-layer tests
python -m pytest
```

In the commands below, `python` is your environment's interpreter
(`.venv/Scripts/python.exe` on Windows). Flags: `--model {xgb,lgb,blend}`
(`blend` by default), `--folds`, `--seed`, `--n-estimators`, `--sample-train N`,
`--submissions-dir`, `--outdir`, `--save-model`, `--model-dir` (details in `train.py --help`).

## Inference

To predict with the saved models (no retraining):

```bash
# 1) train and save the final models into outputs/model/
python train.py --save-model

# 2) predict from any csv with the same columns and write a submission
python predict.py
```

`predict.py --input path/to/file.csv` accepts any csv with the same columns as
`data/raw/test.csv` (the `satisfaction` column is not required). By default it
reads `data/raw/test.csv` and writes `submissions/submission.csv`.

## Structure

```
.
├── train.py            # entry point: CV + OOF AUC + submission (+ --save-model)
├── predict.py          # inference with the saved models from outputs/model/
├── src/
│   ├── data.py         # loading, cleaning, features, encode, inference transform, folds
│   └── model.py        # XGBoost/LightGBM, cv_predict, fit_final (early stopping)
├── tests/
│   └── test_data.py    # data-layer tests
├── data/raw/           # train.csv / test.csv / sample_submission.csv (not in git)
├── outputs/model/      # final models *.joblib + feature_columns.json (not in git)
├── pyproject.toml      # pytest config
├── requirements.txt
├── LICENSE             # MIT license
├── .gitignore          # what stays out of git (data, outputs, .venv…)
├── README.md           # this file
└── README.ru.md        # russian
```

## Results (OOF AUC, 5-fold, measured on the full data)

| Variant | OOF AUC |
|---------|---------|
| XGBoost | 0.9587 |
| LightGBM | 0.9587 |
| **Blend (XGB + LGB, average)** | **0.9588** |

Sample output of `python train.py` (full run; the block is abridged — the
in-between fold lines are omitted and the submission path is shown relative,
in the real log it's absolute):

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

The research ensemble (blend_05 — 146 OOF models, sweeping seeds and weights)
reached **0.96186 on the leaderboard** (OOF 0.9622), but it was cut for the sake
of simplicity: +0.003 AUC at the cost of hundreds of files and hours of compute.
The old code lives on only locally in `research_archive/` (not in git) and plays
no part in the workflow.

## Submission format

`submissions/submission.csv`:

- exactly two columns: `id,satisfaction`;
- `satisfaction` is a probability in `[0,1]` (`predict_proba(...)[:, 1]`), not 0/1 and not strings;
- 299,844 rows, in the same row order as `sample_submission.csv`.
