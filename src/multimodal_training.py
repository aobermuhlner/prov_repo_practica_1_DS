"""Resumable nested-OOF or holdout stacking. Run scripts/train_multimodal.py --help."""
from __future__ import annotations

import argparse
import hashlib
import json
import uuid
import warnings
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.base import BaseEstimator, TransformerMixin, clone
from sklearn.calibration import CalibratedClassifierCV
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.frozen import FrozenEstimator
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import f1_score
from sklearn.model_selection import ParameterGrid, StratifiedKFold, train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler
from threadpoolctl import threadpool_limits
from xgboost import XGBClassifier

ROOT = Path(__file__).resolve().parents[1]
NUMERIC = ["net_profit_margin", "ebitda_margin", "asset_turnover"]
TABULAR = ["country_code", *NUMERIC]


def save(value, path):
    """Atomic checkpoints: an interruption leaves the preceding checkpoint intact."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    joblib.dump(value, temporary)
    temporary.replace(path)


def write_json(value, path):
    path = Path(path)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2, default=str) + "\n")
    temporary.replace(path)


def fingerprint(path):
    digest = hashlib.sha256()
    with open(path, "rb") as source:
        for chunk in iter(lambda: source.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


class FinancialCleaner(TransformerMixin, BaseEstimator):
    """Fold-local cleaning; preserve missingness before replacing values."""
    def __init__(self, imputation="country", clip=False, indicators=True):
        self.imputation = imputation
        self.clip = clip
        self.indicators = indicators

    def fit(self, X, y=None):
        values = X[NUMERIC].replace([np.inf, -np.inf], np.nan)
        self.lower_ = values.quantile(0.005)
        self.upper_ = values.quantile(0.995)
        if self.clip:
            values = values.clip(self.lower_, self.upper_, axis=1)
        self.global_ = values.median().fillna(0.0)
        grouped = values.groupby(X.country_code)
        self.country_ = grouped.median().where(grouped.count() >= 10)
        return self

    def transform(self, X):
        result = X[TABULAR].copy()
        result["country_code"] = result.country_code.fillna("__MISSING__").astype(str)
        for col in NUMERIC:
            raw = X[col]
            values = raw.replace([np.inf, -np.inf], np.nan)
            if self.indicators:
                result[col + "_missing"] = values.isna().astype(float)
                result[col + "_infinite"] = np.isinf(raw).astype(float)
            if self.clip:
                values = values.clip(self.lower_[col], self.upper_[col])
            if self.imputation == "country":
                values = values.fillna(X.country_code.map(self.country_[col]))
            if self.imputation != "native":
                values = values.fillna(self.global_[col])
            result[col] = values
        return result


def numeric_columns(frame):
    return [c for c in frame.columns if c != "country_code"]


def tabular_pipeline(model):
    return Pipeline([
        ("clean", FinancialCleaner()),
        ("prep", ColumnTransformer([
            ("numeric", "passthrough", numeric_columns),
            ("country", OneHotEncoder(handle_unknown="ignore", sparse_output=False), ["country_code"]),
        ], sparse_threshold=0)),
        ("model", model),
    ])


def model_specs(jobs, embedding_hgb=False, quick=False):
    """Small explicit grids: 12 KNN, 8 LR, 12 XGB, 8 RF candidates."""
    specs = {
        "emb_knn": (Pipeline([("scale", "passthrough"), ("model", KNeighborsClassifier(metric="cosine", n_jobs=jobs))]), {
            "scale": ["passthrough", StandardScaler()],
            "model__n_neighbors": [11, 19, 35], "model__weights": ["uniform", "distance"],
        }),
        "emb_lr": (Pipeline([("scale", StandardScaler()), ("model", LogisticRegression(max_iter=1500))]), {
            "scale": ["passthrough", StandardScaler()], "model__C": [0.01, 0.1, 1.0, 10.0],
        }),
        "tab_xgb": (tabular_pipeline(XGBClassifier(
            n_estimators=150, learning_rate=0.08, tree_method="hist", objective="multi:softprob",
            eval_metric="mlogloss", n_jobs=jobs, random_state=42,
        )), {
            "clean__imputation": ["country", "global", "native"],
            "clean__indicators": [False, True], "model__max_depth": [3, 5],
        }),
        "tab_rf": (tabular_pipeline(RandomForestClassifier(n_estimators=200, max_depth=15, n_jobs=jobs, random_state=42)), {
            "clean__imputation": ["country", "global"], "clean__clip": [False, True],
            "model__min_samples_leaf": [2, 8],
        }),
    }
    if embedding_hgb:
        specs["emb_hgb"] = (Pipeline([
            ("pca", PCA(n_components=32, svd_solver="randomized", random_state=42)),
            ("model", HistGradientBoostingClassifier(max_iter=100, early_stopping=False, random_state=42)),
        ]), {"model__max_leaf_nodes": [7, 15], "model__l2_regularization": [1.0, 10.0]})
    if quick:
        specs = {name: (model, {key: [values[0]] for key, values in grid.items()}) for name, (model, grid) in specs.items()}
        for name, (model, _) in specs.items():
            if name.startswith("tab_"):
                model.set_params(model__n_estimators=5)
    return specs


def metrics(y, probabilities, classes):
    predictions = classes[probabilities.argmax(axis=1)]
    confidence = probabilities.max(axis=1)
    correct = predictions == y
    f1 = f1_score(y, predictions, average="weighted", zero_division=0)
    brier = np.mean((confidence - correct) ** 2)
    return {"weighted_f1": float(f1), "confidence_brier": float(brier),
            "combined": float(f1 * (1 - brier)), "accuracy": float(correct.mean()),
            "mean_confidence": float(confidence.mean())}


def evaluate_model(model, X, y, *, calibration_plots=False):
    """Print scores by default; optional confidence table and reliability plot."""
    probabilities = model.predict_proba(X)
    result = metrics(y, probabilities, model.classes_)
    print("  ".join(f"{key}={value:.4f}" for key, value in result.items()), flush=True)
    if calibration_plots:
        import matplotlib.pyplot as plt
        confidence = probabilities.max(axis=1)
        correct = model.classes_[probabilities.argmax(axis=1)] == y
        table = pd.DataFrame({"confidence": confidence, "correct": correct})
        table["bin"] = pd.cut(confidence, np.linspace(0, 1, 11), include_lowest=True)
        table = table.groupby("bin", observed=True).agg(count=("correct", "size"), confidence=("confidence", "mean"), accuracy=("correct", "mean"))
        print(table)
        _, ax = plt.subplots()
        ax.plot(table.confidence, table.accuracy, "o-")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy")
        plt.show()
    return result


def load_data(args):
    train, validation = pd.read_pickle(args.train), pd.read_pickle(args.validation)
    financial = pd.read_pickle(args.financial).groupby("id", as_index=False).first()
    if train.id.duplicated().any() or validation.id.duplicated().any() or set(train.id) & set(validation.id):
        raise ValueError("Company IDs must be unique and disjoint between train and validation")
    train = train.merge(financial, on="id", how="left", validate="one_to_one")
    validation = validation.merge(financial, on="id", how="left", validate="one_to_one")
    data = pd.concat([train, validation], ignore_index=True)
    embeddings = np.asarray(data.business_description_embedding.tolist(), dtype=np.float32)
    if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
        raise ValueError("Embeddings must be a finite 2D array")
    tabular = data[TABULAR].copy()
    tabular[NUMERIC] = tabular[NUMERIC].apply(pd.to_numeric, errors="raise")
    encoder = LabelEncoder().fit(train.industry)
    y = encoder.transform(data.industry)
    return data, embeddings, tabular, y, encoder, len(train)


def create_splits(y, n_train, seed, strategy="oof"):
    # Preserve the previous test membership so observed test rows never migrate
    # into model development when the protocol changes.
    base, meta = train_test_split(
        np.arange(n_train), test_size=0.35, stratify=y[:n_train], random_state=seed,
    )
    meta_fit, meta_select = train_test_split(
        meta, test_size=0.4, stratify=y[meta], random_state=seed,
    )
    heldout = np.arange(n_train, len(y))
    calibration_pool, test = train_test_split(
        heldout, test_size=0.5, stratify=y[heldout], random_state=seed,
    )
    calibration, calibration_select = train_test_split(
        calibration_pool, test_size=0.5, stratify=y[calibration_pool], random_state=seed,
    )
    if strategy == "holdout":
        splits = dict(base=base, meta_fit=meta_fit, meta_select=meta_select)
    else:
        splits = dict(base=np.sort(np.concatenate([base, meta_fit])), meta_select=meta_select)
    return dict(**splits, calibration=calibration, calibration_select=calibration_select, test=test)


def take(X, indices):
    return X.iloc[indices] if hasattr(X, "iloc") else X[indices]


class FitDidNotConverge(RuntimeError):
    def __init__(self, diagnostics):
        super().__init__("Fit did not converge after bounded retries; see convergence diagnostics")
        self.diagnostics = diagnostics


def fit_checked(estimator, X, y):
    """Retry logistic fits without changing features, C, or the objective."""
    estimator = clone(estimator)
    terminal = estimator.steps[-1][1] if isinstance(estimator, Pipeline) else estimator
    logistic = isinstance(terminal, LogisticRegression)
    attempts = [(terminal.solver, terminal.max_iter)] if logistic else [(None, None)]
    if logistic:
        attempts.extend([("lbfgs", max(6000, terminal.max_iter)), ("newton-cg", 2000)])
    records = []
    for solver, limit in attempts:
        fitted = clone(estimator)
        model = fitted.steps[-1][1] if isinstance(fitted, Pipeline) else fitted
        if logistic:
            model.set_params(solver=solver, max_iter=limit)
        with warnings.catch_warnings(record=True) as caught:
            warnings.simplefilter("always", ConvergenceWarning)
            fitted.fit(X, y)
        convergence_warnings = []
        for warning in caught:
            if issubclass(warning.category, ConvergenceWarning):
                convergence_warnings.append(str(warning.message))
            else:
                warnings.warn_explicit(str(warning.message), warning.category, warning.filename, warning.lineno)
        record = {"solver": solver, "max_iter": limit,
                  "n_iter": np.asarray(getattr(model, "n_iter_", [])).tolist(),
                  "converged": not convergence_warnings, "warnings": convergence_warnings}
        records.append(record)
        if record["converged"]:
            return fitted, {"converged": True, "attempts": records}
        print(f"  convergence retry: {solver}, limit={limit}", flush=True)
    raise FitDidNotConverge({"converged": False, "attempts": records})


def checked_oof(estimator, X, y, folds, seed, directory):
    """Checkpoint individual folds; each fit excludes its prediction rows."""
    probabilities = np.empty((len(y), len(np.unique(y))))
    diagnostics = []
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    for fold, (train, valid) in enumerate(cv.split(X, y)):
        path = directory / f"fold_{fold:02d}.joblib"
        if path.exists():
            result = joblib.load(path)
        else:
            fitted, diagnostic = fit_checked(estimator, take(X, train), y[train])
            result = {"valid": valid, "probabilities": fitted.predict_proba(take(X, valid)),
                      "convergence": diagnostic}
            save(result, path)
        np.testing.assert_array_equal(result["valid"], valid)
        probabilities[valid] = result["probabilities"]
        diagnostics.append(result["convergence"])
    return probabilities, diagnostics


def tune_branch(name, model, grid, X, y, directory, folds, seed):
    final = directory / "best.joblib"
    if final.exists():
        print(f"Resume {name}: fitted winner", flush=True)
        return joblib.load(final)
    results = []
    candidates = list(ParameterGrid(grid))
    for i, parameters in enumerate(candidates):
        checkpoint = directory / f"trial_{i:03d}.joblib"
        if checkpoint.exists():
            result = joblib.load(checkpoint)
        else:
            print(f"{name}: trial {i + 1}/{len(candidates)} {parameters}", flush=True)
            candidate = clone(model).set_params(**parameters)
            try:
                proba, diagnostics = checked_oof(
                    candidate, X, y, folds, seed, directory / f"trial_{i:03d}_folds",
                )
                result = {"params": parameters, "eligible": True, "convergence": diagnostics,
                          **metrics(y, proba, np.unique(y))}
            except FitDidNotConverge as error:
                result = {"params": parameters, "eligible": False, "convergence": error.diagnostics}
            save(result, checkpoint)
        results.append(result)
    pd.DataFrame(results).to_csv(directory / "trials.csv", index=False)
    write_json(results, directory / "convergence.json")
    eligible = [row for row in results if row["eligible"]]
    if not eligible:
        raise RuntimeError(f"No converged candidates for {name}; inspect {directory}")
    best = max(eligible, key=lambda row: row["combined"])
    print(f"{name}: refit winner, CV combined={best['combined']:.4f}", flush=True)
    fitted, diagnostic = fit_checked(clone(model).set_params(**best["params"]), X, y)
    write_json({**best, "refit_convergence": diagnostic}, directory / "best_params.json")
    save(fitted, final)
    return fitted


def branch_X(name, indices, embeddings, tabular):
    return embeddings[indices] if name.startswith("emb_") else tabular.iloc[indices]


def meta_features(probabilities, names, representation):
    matrix = np.hstack([probabilities[name] for name in names])
    return np.log(np.clip(matrix, 1e-7, 1)) if representation == "log_proba" else matrix


def meta_estimator(C):
    # In particular, log-probability columns can have very different spreads.
    return Pipeline([("scale", StandardScaler()),
                     ("model", LogisticRegression(C=C, max_iter=1500))])


def tune_meta(training, selection, y_train, y_select, directory):
    path = directory / "meta.joblib"
    if path.exists():
        return joblib.load(path)
    names = list(training)
    embedding_names = [name for name in names if name.startswith("emb_")]
    subsets = [embedding_names, names, [n for n in names if n != "tab_rf"],
               [n for n in names if n != "tab_xgb"], ["emb_knn"]]
    subsets = list(dict.fromkeys(tuple(group) for group in subsets))
    candidates = [(group, representation, C) for group in subsets
                  for representation in ["proba", "log_proba"] for C in [0.01, 0.1, 1., 10.]]
    results = []
    for i, (group, representation, C) in enumerate(candidates):
        checkpoint = directory / "meta_trials" / f"trial_{i:03d}.joblib"
        if checkpoint.exists():
            row = joblib.load(checkpoint)
        else:
            print(f"Meta trial {i + 1}/{len(candidates)}: {group}, {representation}, C={C}", flush=True)
            row = {"names": group, "representation": representation, "C": C}
            try:
                fitted, diagnostic = fit_checked(
                    meta_estimator(C), meta_features(training, group, representation), y_train,
                )
                row.update(eligible=True, convergence=diagnostic,
                           **metrics(y_select, fitted.predict_proba(meta_features(selection, group, representation)), fitted.classes_))
            except FitDidNotConverge as error:
                row.update(eligible=False, convergence=error.diagnostics)
            save(row, checkpoint)
        results.append(row)
    pd.DataFrame(results).to_csv(directory / "meta_trials.csv", index=False)
    write_json(results, directory / "meta_convergence.json")
    eligible = [row for row in results if row["eligible"]]
    if not eligible:
        raise RuntimeError("No converged meta candidates; inspect meta_convergence.json")
    best = max(eligible, key=lambda row: row["combined"])
    winner = {key: best[key] for key in ["names", "representation", "C"]}
    write_json({**winner, "selection_score": best["combined"]}, directory / "meta_best_params.json")
    features = np.vstack([meta_features(part, winner["names"], winner["representation"])
                          for part in [training, selection]])
    fitted, diagnostic = fit_checked(meta_estimator(winner["C"]), features, np.concatenate([y_train, y_select]))
    chosen = {**winner, "model": fitted, "convergence": diagnostic}
    write_json(diagnostic, directory / "meta_refit_convergence.json")
    save(chosen, path)
    return chosen


def cached_predictions(branches, indices, embeddings, tabular, directory, split):
    predictions = {}
    for name, model in branches.items():
        path = directory / "predictions" / f"{split}_{name}.joblib"
        if not path.exists():
            save(model.predict_proba(branch_X(name, indices, embeddings, tabular)), path)
        predictions[name] = joblib.load(path)
    return predictions


def nested_oof(specs, indices, embeddings, tabular, y, directory, folds, seed):
    """Every outer-held-out row is excluded from inner hyperparameter search."""
    cv = StratifiedKFold(folds, shuffle=True, random_state=seed)
    predictions = {name: np.empty((len(indices), len(np.unique(y)))) for name in specs}
    for fold, (train, valid) in enumerate(cv.split(indices, y[indices])):
        fold_dir = directory / "outer_folds" / f"fold_{fold:02d}"
        actual_train, actual_valid = indices[train], indices[valid]
        save({"train": actual_train, "valid": actual_valid}, fold_dir / "indices.joblib")
        for name, (model, grid) in specs.items():
            path = fold_dir / f"{name}_proba.joblib"
            if not path.exists():
                print(f"Outer fold {fold + 1}/{folds}: {name}", flush=True)
                fitted = tune_branch(name, model, grid, branch_X(name, actual_train, embeddings, tabular),
                                     y[actual_train], fold_dir / name, folds, seed)
                save(fitted.predict_proba(branch_X(name, actual_valid, embeddings, tabular)), path)
            predictions[name][valid] = joblib.load(path)
    return predictions


def select_calibration(raw, calibrated, X, y):
    scores = {"raw": metrics(y, raw.predict_proba(X), raw.classes_),
              "temperature": metrics(y, calibrated.predict_proba(X), calibrated.classes_)}
    # A tie keeps the simpler, raw prediction rule.
    method = "temperature" if scores["temperature"]["combined"] > scores["raw"]["combined"] + 1e-12 else "raw"
    return {"method": method, "scores": scores}


def predict_bundle(bundle, embeddings, tabular, *, calibrated=None):
    """Default to the development-selected rule; True/False force temperature/raw."""
    probabilities = {
        name: bundle["branches"][name].predict_proba(embeddings if name.startswith("emb_") else tabular)
        for name in bundle["names"]
    }
    features = meta_features(probabilities, bundle["names"], bundle["representation"])
    if calibrated is None:
        model = bundle.get("selected_meta", bundle["calibrated_meta"])
    else:
        model = bundle["calibrated_meta"] if calibrated else bundle["meta"]
    proba = model.predict_proba(features)
    labels = bundle["label_encoder"].inverse_transform(model.classes_[proba.argmax(axis=1)])
    return proba, labels


def prepare_run(args):
    excluded = {"retrain", "through", "evaluate_test", "calibration_plots", "compare_calibration", "output"}
    config = {key: value for key, value in vars(args).items() if key not in excluded}
    signature = {"config": config,
                 "inputs": {str(path): fingerprint(path) for path in [args.train, args.validation, args.financial]},
                 "source": fingerprint(__file__), "sklearn": sklearn.__version__, "numpy": np.__version__,
                 "pandas": pd.__version__, "xgboost": __import__("xgboost").__version__}
    if args.reuse_base_from:
        signature["reuse_inputs"] = {str(p): fingerprint(p) for p in sorted(args.reuse_base_from.rglob("best.joblib"))}
    signature = json.loads(json.dumps(signature, default=str))
    args.output.mkdir(parents=True, exist_ok=True)
    path = args.output / "manifest.json"
    if path.exists() and not args.retrain:
        manifest = json.loads(path.read_text())
        if manifest["signature"] != signature:
            raise ValueError("Data/code/config changed. Use --retrain or a new --output; stale artifacts are not reused.")
    else:
        manifest = {"run": uuid.uuid4().hex[:12], "signature": signature, "protocol": 2}
        write_json(manifest, path)
    directory = args.output / manifest["run"]
    directory.mkdir(exist_ok=True)
    write_json(manifest, directory / "manifest.json")
    print(f"Run directory: {directory}", flush=True)
    return directory, signature


def reuse_branches(args, splits, specs, signature, directory):
    source = args.reuse_base_from
    manifest_path = source / "manifest.json"
    if not manifest_path.exists():
        manifest_path = source.parent / "manifest.json"  # Original protocol-1 layout.
    manifest = json.loads(manifest_path.read_text())
    if manifest["run"] != source.name:
        raise ValueError("Source run metadata does not match --reuse-base-from")
    previous = manifest["signature"]
    if previous["config"].get("strategy", "holdout") != "holdout":
        raise ValueError("--reuse-base-from needs a holdout run; OOF artifacts must resume in their own run")
    for key in ["inputs", "sklearn", "numpy", "pandas", "xgboost"]:
        if previous[key] != signature[key]:
            raise ValueError(f"Cannot reuse branches: source {key} differs")
    for key in ["seed", "folds", "embedding_hgb", "quick"]:
        if previous["config"][key] != signature["config"][key]:
            raise ValueError(f"Cannot reuse branches: source {key} differs")
    old_splits = joblib.load(source / "splits.joblib")
    for key in ["base", "meta_fit", "meta_select", "test"]:
        np.testing.assert_array_equal(old_splits[key], splits[key])
    old_calibration = np.concatenate([
        old_splits["calibration"], old_splits.get("calibration_select", np.array([], dtype=int)),
    ])
    new_calibration = np.concatenate([splits["calibration"], splits["calibration_select"]])
    np.testing.assert_array_equal(np.sort(old_calibration), np.sort(new_calibration))
    branches = {}
    for name in specs:
        path = source / name / "best.joblib"
        if not path.exists():
            path = source / "search" / name / "best.joblib"
        branches[name] = joblib.load(path)
        print(f"Reuse fitted branch: {name}", flush=True)
    write_json({"source": str(source), "source_signature": previous}, directory / "reused_branches.json")
    return branches


def run(args):
    directory, signature = prepare_run(args)
    data, embeddings, tabular, y, encoder, n_train = load_data(args)
    split_path = directory / "splits.joblib"
    if split_path.exists():
        splits = joblib.load(split_path)
    else:
        splits = create_splits(y, n_train, args.seed, args.strategy)
        save(splits, split_path)
        write_json({name: data.iloc[ix].id.tolist() for name, ix in splits.items()}, directory / "split_ids.json")
        save(encoder, directory / "label_encoder.joblib")
    for name, ix in splits.items():
        print(f"{name}: {len(ix)} rows", flush=True)
        if len(np.unique(y[ix])) != len(encoder.classes_):
            raise ValueError(f"Split {name} lacks classes; need more examples or different split sizes")
    minimum = np.bincount(y[splits['base']]).min()
    if minimum < args.folds or (args.strategy == "oof" and minimum - int(np.ceil(minimum / args.folds)) < args.folds):
        raise ValueError("Insufficient examples per class for the requested inner/outer CV folds")
    if args.through == "splits":
        return
    specs = model_specs(args.jobs, args.embedding_hgb, args.quick)
    if args.reuse_base_from:
        branches = reuse_branches(args, splits, specs, signature, directory)
    else:
        branches = {name: tune_branch(
            name, model, grid, branch_X(name, splits["base"], embeddings, tabular), y[splits["base"]],
            directory / "search" / name, args.folds, args.seed,
        ) for name, (model, grid) in specs.items()}
    if args.through == "base":
        return
    selection = cached_predictions(branches, splits["meta_select"], embeddings, tabular, directory, "meta_select")
    if args.strategy == "oof":
        training = nested_oof(specs, splits["base"], embeddings, tabular, y, directory, args.folds, args.seed)
        training_y = y[splits["base"]]
    else:
        training = cached_predictions(branches, splits["meta_fit"], embeddings, tabular, directory, "meta_fit")
        training_y = y[splits["meta_fit"]]
    development = {name: metrics(y[splits["meta_select"]], selection[name], model.classes_)
                   for name, model in branches.items()}
    print("Individual development scores (meta_select):\n" + pd.DataFrame(development).T.to_string(), flush=True)
    pd.DataFrame(development).T.to_csv(directory / "individual_development_scores.csv")
    chosen = tune_meta(training, selection, training_y, y[splits["meta_select"]], directory)
    print(f"Selected meta-model: {chosen['names']}, {chosen['representation']}, C={chosen['C']}", flush=True)
    baseline = metrics(y[splits['meta_select']], np.mean(list(selection.values()), axis=0), np.unique(y))
    write_json(baseline, directory / "soft_vote_development.json")
    if args.through == "meta":
        return
    if args.strategy == "oof":
        # All choices are now fixed. Final branches use 100% of original train;
        # no calibration or test row enters this refit.
        for name, model in list(branches.items()):
            path = directory / "final_branches" / f"{name}.joblib"
            if path.exists():
                branches[name] = joblib.load(path)
            else:
                print(f"Full-training refit: {name}, {n_train} rows", flush=True)
                branches[name], diagnostic = fit_checked(
                    model, branch_X(name, np.arange(n_train), embeddings, tabular), y[:n_train],
                )
                save(branches[name], path)
                write_json(diagnostic, path.with_suffix(".json"))
        write_json(data.iloc[:n_train].id.tolist(), directory / "final_training_ids.json")
    calibrations, decisions, selected_models = {}, {}, {}
    raw_models = {**branches, "stack": chosen["model"]}
    probability_sets = {split: cached_predictions(branches, splits[split], embeddings, tabular, directory, "final_" + split)
                        for split in ["calibration", "calibration_select"]}
    for name, raw in raw_models.items():
        def features(split):
            if name == "stack":
                return meta_features(probability_sets[split], chosen["names"], chosen["representation"])
            return branch_X(name, splits[split], embeddings, tabular)
        path = directory / f"calibrated_{name}.joblib"
        if path.exists():
            calibrated = joblib.load(path)
        else:
            calibrated = CalibratedClassifierCV(FrozenEstimator(raw), method="temperature").fit(
                features("calibration"), y[splits["calibration"]],
            )
            save(calibrated, path)
        calibrations[name] = calibrated
        decisions[name] = select_calibration(raw, calibrated, features("calibration_select"), y[splits["calibration_select"]])
        selected_models[name] = calibrated if decisions[name]["method"] == "temperature" else raw
        print(f"Calibration selection: {name} -> {decisions[name]['method']}", flush=True)
    write_json(decisions, directory / "calibration_selection.json")
    bundle = {"branches": branches, "names": chosen["names"], "representation": chosen["representation"],
              "meta": chosen["model"], "calibrated_meta": calibrations["stack"], "selected_meta": selected_models["stack"],
              "selected_individuals": {n: selected_models[n] for n in branches},
              "calibration_selection": decisions, "label_encoder": encoder}
    save(bundle, directory / "bundle.joblib")
    if not args.evaluate_test:
        print("Saved selected inference bundle. Test remains unevaluated; use --evaluate-test for final scores.", flush=True)
        return
    test_probabilities = cached_predictions(branches, splits["test"], embeddings, tabular, directory, "final_test")
    rows = {}
    for name, raw in raw_models.items():
        features = (meta_features(test_probabilities, chosen['names'], chosen['representation']) if name == "stack"
                    else branch_X(name, splits["test"], embeddings, tabular))
        variants = {"selected": selected_models[name]}
        if args.compare_calibration:
            variants.update(raw=raw, temperature=calibrations[name])
        for suffix, estimator in variants.items():
            print(f"TEST {name} / {suffix} (selection={decisions[name]['method']})", flush=True)
            rows[f"{name}_{suffix}"] = evaluate_model(estimator, features, y[splits['test']], calibration_plots=args.calibration_plots)
    rows["soft_vote"] = metrics(y[splits['test']], np.mean(list(test_probabilities.values()), axis=0), np.unique(y))
    pd.DataFrame(rows).T.to_csv(directory / "test_scores.csv")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/df_train_preprocessed.pkl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/processed/df_val_preprocessed.pkl")
    parser.add_argument("--financial", type=Path, default=ROOT.parent / "data/df_financials_train.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "models/multimodal_search_v2")
    parser.add_argument("--strategy", choices=["oof", "holdout"], default="oof")
    parser.add_argument("--reuse-base-from", type=Path, help="Reuse fitted holdout branches from a compatible run directory")
    parser.add_argument("--retrain", action="store_true", help="Start a new run, retaining old artifacts")
    parser.add_argument("--through", choices=["splits", "base", "meta", "calibrate"], default="calibrate")
    parser.add_argument("--evaluate-test", action="store_true", help="Report final test scores after all selection")
    parser.add_argument("--compare-calibration", action="store_true", help="Also report raw/temperature test diagnostics; never changes selection")
    parser.add_argument("--calibration-plots", action="store_true")
    parser.add_argument("--embedding-hgb", action="store_true")
    parser.add_argument("--quick", action="store_true", help="One candidate per branch and five trees; smoke testing only")
    parser.add_argument("--folds", type=int, default=3)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.jobs < 1 or args.folds < 2:
        parser.error("--jobs must be >=1 and --folds >=2")
    if args.reuse_base_from and (args.strategy != "holdout" or args.retrain):
        parser.error("--reuse-base-from requires --strategy holdout and cannot be combined with --retrain")
    if args.evaluate_test and args.through != "calibrate":
        parser.error("--evaluate-test requires --through calibrate")
    if args.compare_calibration and not args.evaluate_test:
        parser.error("--compare-calibration requires --evaluate-test")
    with threadpool_limits(limits=args.jobs):
        run(args)
