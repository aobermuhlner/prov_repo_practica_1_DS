"""Small multimodal PyTorch classifier; independent of the stacking run."""
from __future__ import annotations

import argparse
import json
import uuid
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
import torch
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp, softmax
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from threadpoolctl import threadpool_limits
from torch import nn
from torch.nn import functional as F
from torch.utils.data import DataLoader, TensorDataset

from src.multimodal_training import (
    ROOT, NUMERIC, FinancialCleaner, fingerprint, load_data, metrics, save, write_json,
)

LOSSES = ["ce", "ce_brier", "ce_score", "multiclass_brier"]


class NeuralPreprocessor:
    """Train-only ratio imputation/clipping/scaling; explicit missingness flags."""
    def __init__(self, imputation="country", clip=True, embedding_normalization="l2"):
        self.imputation = imputation
        self.clip = clip
        self.embedding_normalization = embedding_normalization

    def fit(self, embeddings, tabular):
        self.embedding_dim_ = embeddings.shape[1]
        self.cleaner_ = FinancialCleaner(imputation=self.imputation, clip=self.clip, indicators=True).fit(tabular)
        cleaned = self.cleaner_.transform(tabular)
        ratios = self.signed_log(cleaned[NUMERIC].to_numpy(float))
        self.ratio_scaler_ = StandardScaler().fit(ratios)
        self.countries_ = {country: i + 2 for i, country in enumerate(sorted(tabular.country_code.dropna().astype(str).unique()))}
        self.embedding_scaler_ = StandardScaler().fit(embeddings) if self.embedding_normalization == "standard" else None
        return self

    @staticmethod
    def signed_log(values):
        return np.sign(values) * np.log1p(np.abs(values))

    def transform(self, embeddings, tabular):
        embeddings = np.asarray(embeddings, dtype=np.float32)
        if embeddings.ndim != 2 or not np.isfinite(embeddings).all():
            raise ValueError("Embeddings must be a finite 2D matrix")
        if embeddings.shape[1] != self.embedding_dim_:
            raise ValueError(f"Expected {self.embedding_dim_} embedding dimensions")
        if len(embeddings) != len(tabular):
            raise ValueError("Embedding and tabular rows must align")
        if self.embedding_scaler_ is not None:
            embeddings = self.embedding_scaler_.transform(embeddings)
        elif self.embedding_normalization == "l2":
            embeddings = embeddings / np.maximum(np.linalg.norm(embeddings, axis=1, keepdims=True), 1e-12)
        cleaned = self.cleaner_.transform(tabular)
        ratios = self.ratio_scaler_.transform(self.signed_log(cleaned[NUMERIC].to_numpy(float)))
        flag_columns = [col for col in cleaned if col.endswith(("_missing", "_infinite"))]
        numeric = np.column_stack([ratios, cleaned[flag_columns].to_numpy(float)]).astype(np.float32)
        # 0: unseen country (zero embedding); 1: missing country (learned).
        country = tabular.country_code.astype("string").map(self.countries_).fillna(0)
        country = country.where(tabular.country_code.notna(), 1).to_numpy(dtype=np.int64)
        if not np.isfinite(numeric).all():
            raise ValueError("Financial preprocessing produced nonfinite values")
        return np.ascontiguousarray(embeddings, dtype=np.float32), numeric, country


class MultimodalMLP(nn.Module):
    def __init__(self, embedding_dim, numeric_dim, n_countries, n_classes,
                 embedding_hidden=128, dropout=0.2):
        super().__init__()
        self.country = nn.Embedding(n_countries, 8, padding_idx=0)
        self.embedding_branch = nn.Sequential(
            nn.Linear(embedding_dim, embedding_hidden), nn.LayerNorm(embedding_hidden),
            nn.ReLU(), nn.Dropout(dropout), nn.Linear(embedding_hidden, 64), nn.ReLU(),
        )
        self.tabular_branch = nn.Sequential(
            nn.Linear(numeric_dim + 8, 32), nn.LayerNorm(32), nn.ReLU(),
            nn.Linear(32, 16), nn.ReLU(),
        )
        self.head = nn.Sequential(
            nn.Linear(64 + 16, 64), nn.ReLU(), nn.Dropout(dropout), nn.Linear(64, n_classes),
        )

    def forward(self, embeddings, numeric, country):
        text = self.embedding_branch(embeddings)
        tabular = self.tabular_branch(torch.cat([numeric, self.country(country)], dim=1))
        return self.head(torch.cat([text, tabular], dim=1))  # logits, not softmax


def loss_components(logits, y):
    probabilities = logits.softmax(dim=1)
    target = F.one_hot(y, num_classes=logits.shape[1]).to(probabilities.dtype)
    confidence, prediction = probabilities.max(dim=1)
    correct = (prediction == y).to(probabilities.dtype).detach()
    confidence_brier = (confidence - correct).square().mean()
    true_support = target.sum(dim=0)
    predicted_support = probabilities.sum(dim=0)
    true_positive = (probabilities * target).sum(dim=0)
    soft_f1 = 2 * true_positive / (true_support + predicted_support).clamp_min(1e-8)
    soft_weighted_f1 = (soft_f1 * true_support / len(y)).sum()
    return {
        "ce": F.cross_entropy(logits, y),
        "confidence_brier": confidence_brier,
        "multiclass_brier": (probabilities - target).square().sum(dim=1).mean(),
        "score_surrogate": 1 - soft_weighted_f1 * (1 - confidence_brier),
    }


def training_loss(logits, y, name, brier_weight=1., score_weight=1.):
    parts = loss_components(logits, y)
    if name == "ce_brier":
        return parts["ce"] + brier_weight * parts["confidence_brier"]
    if name == "ce_score":
        return parts["ce"] + score_weight * parts["score_surrogate"]
    return parts[name]


def choose_device(request):
    if request == "cpu":
        return torch.device("cpu")
    if torch.cuda.is_available():
        try:
            # Availability alone does not guarantee this wheel supports the GPU.
            probe = torch.ones((8, 8), device="cuda", requires_grad=True)
            (probe @ probe).sum().backward()
            return torch.device("cuda")
        except RuntimeError as error:
            if request == "cuda":
                raise RuntimeError("CUDA probe failed. GTX 1060 needs a compatible CUDA 12.6 PyTorch wheel; see README_multimodal_nn.md") from error
            print(f"CUDA probe failed; using CPU: {error}", flush=True)
    elif request == "cuda":
        raise RuntimeError("CUDA requested but unavailable; use --device cpu or install a compatible CUDA build")
    return torch.device("cpu")


def neural_splits(y, n_train, seed):
    train, development = train_test_split(
        np.arange(n_train), test_size=0.2, stratify=y[:n_train], random_state=seed,
    )
    heldout = np.arange(n_train, len(y))
    calibration_pool, test = train_test_split(
        heldout, test_size=0.5, stratify=y[heldout], random_state=seed,
    )
    calibration, calibration_select = train_test_split(
        calibration_pool, test_size=0.5, stratify=y[calibration_pool], random_state=seed,
    )
    return dict(train=train, development=development, calibration=calibration,
                calibration_select=calibration_select, test=test)


def torch_save(value, path):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    torch.save(value, temporary)
    temporary.replace(path)


def cpu_state(model):
    return {name: tensor.detach().cpu().clone() for name, tensor in model.state_dict().items()}


@torch.inference_mode()
def predict_logits(model, arrays, indices, device, batch_size):
    model.eval()
    batches = []
    for start in range(0, len(indices), batch_size):
        ix = indices[start:start + batch_size]
        inputs = [torch.from_numpy(array[ix]).to(device) for array in arrays]
        batches.append(model(*inputs).cpu().numpy())
    return np.concatenate(batches)


def train_candidate(name, config, arrays, y, splits, args, directory, device):
    torch.manual_seed(args.seed)
    if device.type == "cuda":
        torch.cuda.manual_seed_all(args.seed)
    model = MultimodalMLP(**config).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    checkpoint_path = directory / name / "latest.pt"
    start, best_score, bad_epochs, history, best_state, best_epoch = 0, -np.inf, 0, [], None, 0
    if checkpoint_path.exists():
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=True)
        model.load_state_dict(checkpoint["model"])
        optimizer.load_state_dict(checkpoint["optimizer"])
        start, best_score = checkpoint["epoch"], checkpoint["best_score"]
        bad_epochs, history = checkpoint["bad_epochs"], checkpoint["history"]
        best_state, best_epoch = checkpoint["best_state"], checkpoint["best_epoch"]
        torch.set_rng_state(checkpoint["rng_cpu"])
        if device.type == "cuda" and checkpoint["rng_cuda"] is not None:
            torch.cuda.set_rng_state_all(checkpoint["rng_cuda"])
        print(f"Resume {name}: epoch {start}, best epoch {best_epoch}", flush=True)
    train_ix = splits["train"]
    dataset = TensorDataset(*[torch.from_numpy(array[train_ix]) for array in arrays], torch.from_numpy(y[train_ix]))
    for epoch in range(start, args.epochs):
        if bad_epochs >= args.patience:
            break
        generator = torch.Generator().manual_seed(args.seed + epoch)
        loader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, generator=generator,
                            num_workers=0, pin_memory=device.type == "cuda")
        model.train()
        total_loss = 0.
        for embeddings, numeric, country, labels in loader:
            inputs = [item.to(device, non_blocking=device.type == "cuda") for item in [embeddings, numeric, country]]
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)
            loss = training_loss(model(*inputs), labels, name, args.brier_weight, args.score_weight)
            if not torch.isfinite(loss):
                raise RuntimeError(f"Nonfinite {name} loss in epoch {epoch + 1}")
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=5., error_if_nonfinite=True)
            optimizer.step()
            total_loss += loss.item() * len(labels)
        logits = predict_logits(model, arrays, splits["development"], device, args.batch_size)
        result = metrics(y[splits["development"]], softmax(logits, axis=1), np.arange(config["n_classes"]))
        history.append({"epoch": epoch + 1, "training_loss": total_loss / len(train_ix), **result})
        if result["combined"] > best_score + args.min_delta:
            best_score, best_state, best_epoch, bad_epochs = result["combined"], cpu_state(model), epoch + 1, 0
        else:
            bad_epochs += 1
        print(f"{name}: epoch {epoch + 1}/{args.epochs}, loss={total_loss / len(train_ix):.4f}, "
              f"dev F1={result['weighted_f1']:.4f}, Brier={result['confidence_brier']:.4f}, "
              f"combined={result['combined']:.4f} (best={best_score:.4f})", flush=True)
        torch_save({"model": cpu_state(model), "optimizer": optimizer.state_dict(), "epoch": epoch + 1,
                    "best_score": best_score, "bad_epochs": bad_epochs, "history": history,
                    "best_state": best_state, "best_epoch": best_epoch,
                    "rng_cpu": torch.get_rng_state(),
                    "rng_cuda": torch.cuda.get_rng_state_all() if device.type == "cuda" else None}, checkpoint_path)
        pd.DataFrame(history).to_csv(directory / name / "history.csv", index=False)
    if best_state is None:
        raise RuntimeError("No completed training epoch available")
    model.load_state_dict(best_state)
    torch_save({"state_dict": best_state, "config": config}, directory / name / "best.pt")
    return model, {"loss": name, "best_epoch": best_epoch, "combined": best_score,
                   "epochs_completed": len(history), "early_stopped": bad_epochs >= args.patience}


def fit_temperature(logits, y, objective="confidence_brier"):
    """Positive temperature preserves argmax; confidence Brier directly aligns with our score."""
    logits = np.asarray(logits, dtype=np.float64)
    correct = logits.argmax(axis=1) == y
    def loss(log_temperature):
        scaled = logits / np.exp(log_temperature)
        if objective == "log_loss":
            return float(np.mean(logsumexp(scaled, axis=1) - scaled[np.arange(len(y)), y]))
        confidence = softmax(scaled, axis=1).max(axis=1)
        return float(np.mean((confidence - correct) ** 2))
    result = minimize_scalar(loss, bounds=(-4., 4.), method="bounded")
    temperature = float(np.exp(result.x)) if result.success and result.fun < loss(0.) - 1e-12 else 1.
    return temperature


def calibration_choice(logits, y, temperature):
    classes = np.arange(logits.shape[1])
    scores = {"raw": metrics(y, softmax(logits, axis=1), classes),
              "temperature": metrics(y, softmax(logits / temperature, axis=1), classes)}
    method = "temperature" if scores["temperature"]["combined"] > scores["raw"]["combined"] + 1e-12 else "raw"
    return method, scores


def report(y, logits, temperature=1., plots=False):
    probabilities = softmax(logits / temperature, axis=1)
    result = metrics(y, probabilities, np.arange(logits.shape[1]))
    print("  ".join(f"{key}={value:.4f}" for key, value in result.items()), flush=True)
    if plots:
        import matplotlib.pyplot as plt
        confidence = probabilities.max(axis=1)
        table = pd.DataFrame({"confidence": confidence, "correct": probabilities.argmax(axis=1) == y})
        table["bin"] = pd.cut(confidence, np.linspace(0, 1, 11), include_lowest=True)
        table = table.groupby("bin", observed=True).agg(count=("correct", "size"), confidence=("confidence", "mean"), accuracy=("correct", "mean"))
        print(table)
        _, ax = plt.subplots()
        ax.plot(table.confidence, table.accuracy, "o-")
        ax.plot([0, 1], [0, 1], "--", color="gray")
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="Mean confidence", ylabel="Accuracy")
        plt.show()
    return result


def predict_nn(run_directory, embeddings, tabular, *, device="cpu", batch_size=512, calibrated=None):
    """Predict using saved preprocessing, weights and development-selected temperature."""
    directory = Path(run_directory)
    artifact = joblib.load(directory / "preprocessing.joblib")
    info = json.loads((directory / "inference.json").read_text())
    checkpoint = torch.load(directory / "model.pt", map_location="cpu", weights_only=True)
    target_device = choose_device(device)
    model = MultimodalMLP(**checkpoint["config"]).to(target_device)
    model.load_state_dict(checkpoint["state_dict"])
    arrays = artifact["preprocessor"].transform(embeddings, tabular)
    logits = predict_logits(model, arrays, np.arange(len(embeddings)), target_device, batch_size)
    use_temperature = info["selected_method"] == "temperature" if calibrated is None else calibrated
    probabilities = softmax(logits / (info["temperature"] if use_temperature else 1.), axis=1)
    labels = artifact["label_encoder"].inverse_transform(probabilities.argmax(axis=1))
    return probabilities, labels


def run(args):
    device = choose_device(args.device)
    print(f"Device: {device}; torch {torch.__version__}", flush=True)
    excluded = {"output", "retrain", "epochs", "device", "threads", "evaluate_test", "calibration_plots", "compare_calibration"}
    signature = {"config": {key: value for key, value in vars(args).items() if key not in excluded},
                 "inputs": {str(path): fingerprint(path) for path in [args.train, args.validation, args.financial]},
                 "source": fingerprint(__file__), "shared_source": fingerprint(ROOT / "src/multimodal_training.py"),
                 "torch": str(torch.__version__).split("+")[0], "numpy": np.__version__,
                 "pandas": pd.__version__, "sklearn": sklearn.__version__}
    signature = json.loads(json.dumps(signature, default=str))
    args.output.mkdir(parents=True, exist_ok=True)
    manifest_path = args.output / "manifest.json"
    if manifest_path.exists() and not args.retrain:
        manifest = json.loads(manifest_path.read_text())
        if manifest["signature"] != signature:
            raise ValueError("NN data/code/config changed; use --retrain or a new --output")
    else:
        manifest = {"run": uuid.uuid4().hex[:12], "signature": signature}
        write_json(manifest, manifest_path)
    directory = args.output / manifest["run"]
    directory.mkdir(exist_ok=True)
    write_json(manifest, directory / "manifest.json")
    write_json({"torch": str(torch.__version__), "cuda_build": torch.version.cuda,
                "device": str(device), "epochs_requested": args.epochs, "threads": args.threads,
                "gpu": torch.cuda.get_device_name(0) if device.type == "cuda" else None},
               directory / "runtime.json")
    print(f"Run directory: {directory}", flush=True)
    data, embeddings, tabular, y, encoder, n_train = load_data(args)
    split_path = directory / "splits.joblib"
    if split_path.exists():
        splits = joblib.load(split_path)
    else:
        splits = neural_splits(y, n_train, args.seed)
        save(splits, split_path)
        write_json({name: data.iloc[ix].id.tolist() for name, ix in splits.items()}, directory / "split_ids.json")
    for name, ix in splits.items():
        print(f"{name}: {len(ix)} rows", flush=True)
        if len(np.unique(y[ix])) != len(encoder.classes_):
            raise ValueError(f"Split {name} lacks classes; need more data or different split sizes")
    preprocessing_path = directory / "preprocessing.joblib"
    if preprocessing_path.exists():
        preprocessor = joblib.load(preprocessing_path)["preprocessor"]
    else:
        preprocessor = NeuralPreprocessor(args.imputation, not args.no_clip, args.embedding_normalization)
        ix = splits["train"]
        preprocessor.fit(embeddings[ix], tabular.iloc[ix])
        save({"preprocessor": preprocessor, "label_encoder": encoder}, preprocessing_path)
    arrays = preprocessor.transform(embeddings, tabular)
    config = {"embedding_dim": arrays[0].shape[1], "numeric_dim": arrays[1].shape[1],
              "n_countries": len(preprocessor.countries_) + 2, "n_classes": len(encoder.classes_),
              "embedding_hidden": args.embedding_hidden, "dropout": args.dropout}
    parameter_count = sum(p.numel() for p in MultimodalMLP(**config).parameters())
    write_json({**config, "parameter_count": parameter_count}, directory / "architecture.json")
    print(f"Trainable parameters: {parameter_count:,}", flush=True)
    results = []
    for name in LOSSES if args.loss == "all" else [args.loss]:
        _, result = train_candidate(name, config, arrays, y, splits, args, directory, device)
        results.append(result)
    pd.DataFrame(results).to_csv(directory / "development_scores.csv", index=False)
    winner = max(results, key=lambda result: result["combined"])
    checkpoint = torch.load(directory / winner["loss"] / "best.pt", map_location="cpu", weights_only=True)
    model = MultimodalMLP(**config).to(device)
    model.load_state_dict(checkpoint["state_dict"])
    torch_save(checkpoint, directory / "model.pt")
    print(f"Selected loss={winner['loss']}, epoch={winner['best_epoch']}, dev combined={winner['combined']:.4f}", flush=True)
    fit_logits = predict_logits(model, arrays, splits["calibration"], device, args.batch_size)
    temperature = fit_temperature(fit_logits, y[splits["calibration"]], args.temperature_objective)
    selection_logits = predict_logits(model, arrays, splits["calibration_select"], device, args.batch_size)
    method, calibration_scores = calibration_choice(selection_logits, y[splits["calibration_select"]], temperature)
    info = {**winner, "parameter_count": parameter_count, "selected_method": method,
            "temperature": temperature, "temperature_objective": args.temperature_objective,
            "calibration_selection_scores": calibration_scores}
    write_json(info, directory / "inference.json")
    print(f"Calibration selection: {method}; fitted temperature={temperature:.4f}", flush=True)
    if args.evaluate_test:
        test_logits = predict_logits(model, arrays, splits["test"], device, args.batch_size)
        variants = {"selected": temperature if method == "temperature" else 1.}
        if args.compare_calibration:
            variants.update(raw=1., temperature=temperature)
        scores = {}
        for name, value in variants.items():
            print(f"TEST NN / {name}", flush=True)
            scores[name] = report(y[splits["test"]], test_logits, value, args.calibration_plots)
        pd.DataFrame(scores).T.to_csv(directory / "test_scores.csv")
    else:
        print("Saved NN and preprocessing. Test unevaluated; use --evaluate-test when ready.", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train", type=Path, default=ROOT / "data/processed/df_train_preprocessed.pkl")
    parser.add_argument("--validation", type=Path, default=ROOT / "data/processed/df_val_preprocessed.pkl")
    parser.add_argument("--financial", type=Path, default=ROOT.parent / "data/df_financials_train.pkl")
    parser.add_argument("--output", type=Path, default=ROOT / "models/multimodal_nn")
    parser.add_argument("--loss", choices=[*LOSSES, "confidence_brier", "all"], default="ce_brier")
    parser.add_argument("--epochs", type=int, default=60)
    parser.add_argument("--patience", type=int, default=8)
    parser.add_argument("--min-delta", type=float, default=1e-4)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight-decay", type=float, default=1e-3)
    parser.add_argument("--dropout", type=float, default=0.2)
    parser.add_argument("--embedding-hidden", type=int, default=128)
    parser.add_argument("--embedding-normalization", choices=["l2", "standard", "none"], default="l2")
    parser.add_argument("--imputation", choices=["country", "global"], default="country")
    parser.add_argument("--no-clip", action="store_true")
    parser.add_argument("--brier-weight", type=float, default=1.)
    parser.add_argument("--score-weight", type=float, default=1.)
    parser.add_argument("--temperature-objective", choices=["confidence_brier", "log_loss"], default="confidence_brier")
    parser.add_argument("--device", choices=["auto", "cpu", "cuda"], default="auto")
    parser.add_argument("--threads", type=int, default=2)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--retrain", action="store_true")
    parser.add_argument("--evaluate-test", action="store_true")
    parser.add_argument("--calibration-plots", action="store_true")
    parser.add_argument("--compare-calibration", action="store_true")
    args = parser.parse_args()
    if min(args.epochs, args.patience, args.batch_size, args.threads, args.embedding_hidden) < 1:
        parser.error("epochs, patience, batch size, threads and hidden size must be positive")
    if args.lr <= 0 or min(args.weight_decay, args.brier_weight, args.score_weight, args.min_delta) < 0 or not 0 <= args.dropout < 1:
        parser.error("Invalid optimizer, loss weight, dropout or minimum improvement setting")
    if args.compare_calibration and not args.evaluate_test:
        parser.error("--compare-calibration requires --evaluate-test")
    torch.set_num_threads(args.threads)
    with threadpool_limits(limits=args.threads):
        run(args)
