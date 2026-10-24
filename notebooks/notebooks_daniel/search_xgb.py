"""Bayesian hyperparameter search for the industry XGBoost classifier.

Loads the preprocessed train/val pickles, runs the same BayesSearchCV as
optional.ipynb, scores the refit model on the validation set, and writes
the model plus a results file under models/.

Run it inside screen so it survives an SSH disconnect:

    screen -S xgb-search
    uv run python notebooks/notebooks_daniel/search_xgb.py
    # detach with Ctrl-A then D; reattach with: screen -r xgb-search
"""

import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, f1_score
from sklearn.preprocessing import LabelEncoder
from skopt import BayesSearchCV
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data" / "processed"
OUTPUT_DIR = SCRIPT_DIR / "models"

N_ITER = 10
CV = 3
RANDOM_STATE = 42

SEARCH_SPACES = {
    "learning_rate": (0.01, 1.0, "log-uniform"),
    "n_estimators": (50, 250),
    "max_depth": (1, 50),
    "subsample": (0.1, 1.0, "uniform"),
    "colsample_bytree": (0.1, 1.0, "uniform"),
}


def score_model(y_true, y_pred):
    scores = {
        "accuracy": accuracy_score(y_true, y_pred),
        "f1_macro": f1_score(y_true, y_pred, average="macro"),
        "f1_micro": f1_score(y_true, y_pred, average="micro"),
        "f1_weighted": f1_score(y_true, y_pred, average="weighted"),
    }
    for name, value in scores.items():
        print(f"{name}: {value:.4f}", flush=True)
    return scores


def confidence_score(y_true, y_pred, confidence):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=float)

    f1 = f1_score(y_true, y_pred, average="weighted")
    correct = (y_true == y_pred).astype(float)
    brier = np.mean((confidence - correct) ** 2)
    return float(f1 * (1 - brier))


def to_builtin(value):
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, dict):
        return {key: to_builtin(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_builtin(item) for item in value]
    return value


def load_split(path, label_encoder):
    frame = pd.read_pickle(path)
    features = np.array(frame["business_description_embedding"].tolist())
    labels = label_encoder.transform(frame.industry)
    return features, labels


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    log_path = OUTPUT_DIR / "xgb_search_log.txt"

    print(f"Loading data from {DATA_DIR}", flush=True)
    train_frame = pd.read_pickle(DATA_DIR / "df_train_preprocessed.pkl")
    label_encoder = LabelEncoder().fit(train_frame.industry)
    X_train = np.array(train_frame["business_description_embedding"].tolist())
    y_train = label_encoder.transform(train_frame.industry)
    X_val, y_val = load_split(DATA_DIR / "df_val_preprocessed.pkl", label_encoder)
    print(
        f"Train: {X_train.shape}  Val: {X_val.shape}  Classes: {len(label_encoder.classes_)}",
        flush=True,
    )

    # One thread per fit. The search itself uses every core via n_jobs=-1.
    estimator = XGBClassifier(
        eval_metric="logloss",
        n_jobs=1,
        device="cuda",
        tree_method="hist",
        random_state=RANDOM_STATE,
    )
    search = BayesSearchCV(
        estimator=estimator,
        search_spaces=SEARCH_SPACES,
        n_iter=N_ITER,
        cv=CV,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    def log_iteration(result):
        scores = -np.asarray(result.func_vals, dtype=float)
        message = (
            f"iter {len(scores)}/{N_ITER}  "
            f"cv_accuracy={scores[-1]:.4f}  best={scores.max():.4f}"
        )
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as log_file:
            log_file.write(message + "\n")

    print(f"Starting search ({N_ITER} iterations, {CV}-fold CV)", flush=True)
    started = time.time()
    search.fit(X_train, y_train, callback=log_iteration)
    elapsed = time.time() - started

    best_params = to_builtin(dict(search.best_params_))
    print(f"Best hyperparameters: {best_params}", flush=True)
    print(f"Best CV accuracy: {search.best_score_:.4f}", flush=True)
    print(f"Search time: {elapsed:.1f} seconds", flush=True)

    model = search.best_estimator_
    model_path = OUTPUT_DIR / "xgb_model.json"
    model.save_model(model_path)
    print(f"Saved model to {model_path}", flush=True)

    y_probs = model.predict_proba(X_val)
    y_pred = model.classes_[np.argmax(y_probs, axis=1)]
    y_confidence = np.max(y_probs, axis=1)
    weighted = confidence_score(y_val, y_pred, y_confidence)
    print(f"Validation confidence score: {weighted:.4f}", flush=True)
    validation_scores = score_model(y_val, y_pred)
    validation_scores["confidence_score"] = weighted

    results = {
        "best_params": best_params,
        "best_cv_accuracy": float(search.best_score_),
        "search_seconds": elapsed,
        "validation": validation_scores,
        "classes": label_encoder.classes_.tolist(),
    }
    results_path = OUTPUT_DIR / "xgb_search_results.json"
    results_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
    print(f"Saved results to {results_path}", flush=True)


if __name__ == "__main__":
    main()
