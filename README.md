# Practica 1 – Industry Classification

Data Science project (UZH, Semester 1): predict a company's industry from embeddings of its business description.

## Project structure

```
├── data/
│   ├── raw/            # original data (df_train.pkl, df_financials_train.pkl) – not committed
│   └── processed/      # cleaned / intermediate data – not committed
├── notebooks/
│   ├── 01_industry_classification.ipynb   # main working notebook (from the course skeleton)
│   └── examples/       # course reference notebooks (classification, embeddings, p-values)
├── src/                # reusable Python code (helpers, feature engineering, …)
├── models/             # saved models – not committed
├── reports/
│   └── figures/        # plots for the report / presentation
├── requirements.txt
└── README.md
```

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (macOS/Linux: source .venv/bin/activate)
pip install -r requirements.txt
```

Put the provided data files into `data/raw/`. Notebooks load data via relative paths (`../data/raw/...`), so run Jupyter with the notebook's folder as working directory (the default).

## Conventions

- New notebooks: prefix with a number, e.g. `02_eda.ipynb`, `03_model_tuning.ipynb`.
- Move code you reuse across notebooks into `src/` and import it.
- Clear large outputs before committing notebooks to keep diffs small.
