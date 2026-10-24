"""Compare raw description embeddings with LDA and PCA features.

LDA keeps at most one direction fewer than the number of industries (24 here).
Several industries have fewer training rows than the 768 embedding dimensions,
so an unregularized per-class covariance is singular. QDA is therefore fit
with shrinkage on the raw embeddings, and without shrinkage only after the
data have been reduced below the smallest class count.
"""

import json
import time
from pathlib import Path

import colorcet as cc
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, silhouette_score
from sklearn.preprocessing import LabelEncoder
from xgboost import XGBClassifier

SCRIPT_DIR = Path(__file__).resolve().parent
DATA_DIR = SCRIPT_DIR.parents[1] / "data" / "processed"
FIG_DIR = SCRIPT_DIR / "figures"
RANDOM_STATE = 42
# Smallest training class has 160 rows, so a full-rank class covariance
# needs fewer than 160 columns.
QDA_SAFE_COMPONENTS = (24, 50, 80)


def score_predictions(y_true, y_pred, confidence):
    f1_weighted = f1_score(y_true, y_pred, average="weighted")
    correct = (y_true == y_pred).astype(float)
    brier = float(np.mean((confidence - correct) ** 2))
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "f1_macro": float(f1_score(y_true, y_pred, average="macro")),
        "f1_weighted": f1_weighted,
        "confidence_score": float(f1_weighted * (1.0 - brier)),
    }


def load_split(path, label_encoder):
    frame = pd.read_pickle(path)
    features = np.asarray(frame["business_description_embedding"].tolist(), dtype=np.float64)
    labels = label_encoder.transform(frame.industry)
    return features, labels


def make_logreg():
    return LogisticRegression(max_iter=1000, random_state=RANDOM_STATE)


def make_xgb():
    return XGBClassifier(
        eval_metric="logloss",
        tree_method="hist",
        random_state=RANDOM_STATE,
        learning_rate=0.3,
        n_estimators=200,
        max_depth=6,
        subsample=0.2,
        colsample_bytree=0.2,
        n_jobs=-1,
    )


def evaluate(name, model, features_train, labels_train, features_val, labels_val):
    print(f"Fitting {name}", flush=True)
    started = time.time()
    try:
        model.fit(features_train, labels_train)
        probabilities = model.predict_proba(features_val)
        predicted = model.classes_[np.argmax(probabilities, axis=1)]
        confidence = np.max(probabilities, axis=1)
        scores = score_predictions(labels_val, predicted, confidence)
    except Exception as exc:
        elapsed = time.time() - started
        print(f"  failed after {elapsed:.1f}s: {exc}", flush=True)
        return {"name": name, "seconds": elapsed, "error": str(exc)}

    elapsed = time.time() - started
    scores["name"] = name
    scores["seconds"] = elapsed
    scores["n_features"] = int(features_train.shape[1])
    print(
        "  "
        f"acc={scores['accuracy']:.4f}  "
        f"f1_macro={scores['f1_macro']:.4f}  "
        f"f1_w={scores['f1_weighted']:.4f}  "
        f"conf={scores['confidence_score']:.4f}  "
        f"({elapsed:.1f}s)",
        flush=True,
    )
    return scores


def plot_explained_variance(ratios, path):
    cumulative = np.cumsum(ratios)
    components = np.arange(1, len(ratios) + 1)
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(components, ratios, color="#4C78A8", label="Share of between-class variance")
    ax.plot(components, cumulative, color="#F58518", marker="o", label="Cumulative")
    ax.set_xlabel("LDA component")
    ax.set_ylabel("Explained variance ratio")
    ax.set_title("How many LDA directions the industries actually use")
    ax.set_xticks(components)
    ax.set_ylim(0, 1.05)
    ax.legend()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def plot_projection(projected, labels, class_names, path, title, xlabel, ylabel):
    colors = cc.glasbey[: len(class_names)]
    fig, ax = plt.subplots(figsize=(12, 8))
    for class_index, name in enumerate(class_names):
        mask = labels == class_index
        ax.scatter(
            projected[mask, 0],
            projected[mask, 1],
            s=8,
            alpha=0.55,
            c=colors[class_index],
            label=name,
            linewidths=0,
        )
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.legend(loc="center left", bbox_to_anchor=(1.02, 0.5), fontsize=7, markerscale=2)
    fig.tight_layout()
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)


def plot_scores(results, path):
    kept = [row for row in results if "accuracy" in row]
    names = [row["name"] for row in kept]
    accuracy = [row["accuracy"] for row in kept]
    f1_macro = [row["f1_macro"] for row in kept]
    y_pos = np.arange(len(names))
    fig, ax = plt.subplots(figsize=(10, max(4.5, 0.38 * len(names))))
    ax.barh(y_pos - 0.18, accuracy, height=0.36, color="#4C78A8", label="Accuracy")
    ax.barh(y_pos + 0.18, f1_macro, height=0.36, color="#F58518", label="F1 macro")
    ax.set_yticks(y_pos, names)
    ax.set_xlim(0, 1)
    ax.set_xlabel("Validation score")
    ax.set_title("Validation scores by feature set")
    ax.legend(loc="lower right")
    ax.invert_yaxis()
    fig.tight_layout()
    fig.savefig(path, dpi=150)
    plt.close(fig)


def separability(projected, labels):
    return float(silhouette_score(projected, labels, metric="euclidean"))


def main():
    FIG_DIR.mkdir(parents=True, exist_ok=True)
    train_frame = pd.read_pickle(DATA_DIR / "df_train_preprocessed.pkl")
    label_encoder = LabelEncoder().fit(train_frame.industry)
    class_names = label_encoder.classes_.tolist()
    features_train = np.asarray(train_frame["business_description_embedding"].tolist(), dtype=np.float64)
    labels_train = label_encoder.transform(train_frame.industry)
    features_val, labels_val = load_split(DATA_DIR / "df_val_preprocessed.pkl", label_encoder)

    class_counts = np.bincount(labels_train)
    max_lda = len(class_names) - 1
    print(
        f"Train {features_train.shape}  Val {features_val.shape}  "
        f"classes {len(class_names)}  smallest class {int(class_counts.min())}",
        flush=True,
    )
    print(
        f"Embedding std: min {features_train.std(axis=0).min():.4f}  "
        f"max {features_train.std(axis=0).max():.4f}",
        flush=True,
    )

    print("Fitting LDA and PCA projections", flush=True)
    lda = LinearDiscriminantAnalysis(n_components=max_lda)
    lda.fit(features_train, labels_train)
    train_lda = lda.transform(features_train)
    val_lda = lda.transform(features_val)
    ratios = np.asarray(lda.explained_variance_ratio_, dtype=float)
    cumulative = np.cumsum(ratios)
    for cutoff in (0.5, 0.8, 0.9, 0.95):
        needed = int(np.searchsorted(cumulative, cutoff) + 1)
        print(f"  {cutoff:.0%} of LDA variance needs {needed} components", flush=True)

    pca_components = max(QDA_SAFE_COMPONENTS)
    pca = PCA(n_components=pca_components, random_state=RANDOM_STATE)
    pca.fit(features_train)
    train_pca = pca.transform(features_train)
    val_pca = pca.transform(features_val)
    pca_cumulative = np.cumsum(pca.explained_variance_ratio_)
    print(
        f"  PCA {pca_components} components keep {pca_cumulative[-1]:.1%} of variance",
        flush=True,
    )

    plot_explained_variance(ratios, FIG_DIR / "lda_explained_variance.png")
    plot_projection(
        val_lda[:, :2],
        labels_val,
        class_names,
        FIG_DIR / "lda_val_2d.png",
        "Validation set in the first two LDA directions",
        "LD1",
        "LD2",
    )
    plot_projection(
        val_pca[:, :2],
        labels_val,
        class_names,
        FIG_DIR / "pca_val_2d.png",
        "Validation set in the first two PCA directions",
        "PC1",
        "PC2",
    )
    print(
        f"Silhouette on validation, 2D:  "
        f"LDA {separability(val_lda[:, :2], labels_val):.3f}  "
        f"PCA {separability(val_pca[:, :2], labels_val):.3f}",
        flush=True,
    )

    results = []
    results.append(
        evaluate("LDA classifier", lda, features_train, labels_train, features_val, labels_val)
    )
    results.append(
        evaluate(
            "LDA shrinkage",
            LinearDiscriminantAnalysis(solver="lsqr", shrinkage="auto"),
            features_train,
            labels_train,
            features_val,
            labels_val,
        )
    )
    results.append(
        evaluate("LogReg raw 768", make_logreg(), features_train, labels_train, features_val, labels_val)
    )
    results.append(
        evaluate("XGB raw 768", make_xgb(), features_train, labels_train, features_val, labels_val)
    )

    for n_components in (2, 8, 16, max_lda):
        results.append(
            evaluate(
                f"LogReg LDA-{n_components}",
                make_logreg(),
                train_lda[:, :n_components],
                labels_train,
                val_lda[:, :n_components],
                labels_val,
            )
        )
    for n_components in (24, 80):
        results.append(
            evaluate(
                f"LogReg PCA-{n_components}",
                make_logreg(),
                train_pca[:, :n_components],
                labels_train,
                val_pca[:, :n_components],
                labels_val,
            )
        )

    results.append(
        evaluate("XGB LDA-24", make_xgb(), train_lda[:, :max_lda], labels_train, val_lda[:, :max_lda], labels_val)
    )
    for n_components in (24, 80):
        results.append(
            evaluate(
                f"XGB PCA-{n_components}",
                make_xgb(),
                train_pca[:, :n_components],
                labels_train,
                val_pca[:, :n_components],
                labels_val,
            )
        )

    for reg_param in (0.5, 0.9):
        results.append(
            evaluate(
                f"QDA raw reg={reg_param}",
                QuadraticDiscriminantAnalysis(reg_param=reg_param),
                features_train,
                labels_train,
                features_val,
                labels_val,
            )
        )
    for n_components in QDA_SAFE_COMPONENTS:
        results.append(
            evaluate(
                f"QDA PCA-{n_components}",
                QuadraticDiscriminantAnalysis(),
                train_pca[:, :n_components],
                labels_train,
                val_pca[:, :n_components],
                labels_val,
            )
        )
    results.append(
        evaluate(
            "QDA LDA-24",
            QuadraticDiscriminantAnalysis(),
            train_lda,
            labels_train,
            val_lda,
            labels_val,
        )
    )

    plot_scores(results, FIG_DIR / "lda_qda_scores.png")
    payload = {
        "lda_explained_variance_ratio": ratios.tolist(),
        "pca_explained_variance_ratio_80": pca.explained_variance_ratio_.tolist(),
        "smallest_class": int(class_counts.min()),
        "results": results,
    }
    results_path = FIG_DIR / "lda_qda_results.json"
    results_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(f"Saved figures and {results_path}", flush=True)


if __name__ == "__main__":
    main()
