import ast

import colorcet as cc
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.colors import ListedColormap
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, f1_score
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder, OneHotEncoder

np.random.seed(42)


def score_model(y_true, y_pred, print_results=True):
    # Accuracy
    accuracy = accuracy_score(y_true, y_pred)
    # F1 score
    f1_macro = f1_score(y_true, y_pred, average="macro")
    f1_micro = f1_score(y_true, y_pred, average="micro")
    f1_weighted = f1_score(y_true, y_pred, average="weighted")

    scores = {
        "accuracy": accuracy,
        "f1_macro": f1_macro,
        "f1_micro": f1_micro,
        "f1_weighted": f1_weighted,
    }

    if print_results:
        print(f"Accuracy: {accuracy:.4f}")
        print(f"F1 macro: {f1_macro:.4f}")
        print(f"F1 micro: {f1_micro:.4f}")
        print(f"F1 weighted: {f1_weighted:.4f}")

    return scores


def confidence_score(y_true, y_pred, confidence):
    """
    Score for the confidence weighted prediction task.

    Parameters:
    -----------
    y_true : array-like
        True class labels
    y_pred : array-like
        Predicted class labels
    confidence : array-like
        Confidence for each prediction, in [0, 1]
        (= estimated probability that the prediction is correct)

    Returns:
    --------
    float
        weighted F1 * (1 - Brier score), between 0 and 1 (higher is better)
    """
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    confidence = np.asarray(confidence, dtype=float)

    assert len(y_true) == len(y_pred) == len(confidence), "Inputs must have the same length"
    assert np.all((confidence >= 0) & (confidence <= 1)), "Confidence must be between 0 and 1"

    f1 = f1_score(y_true, y_pred, average="weighted")
    correct = (y_true == y_pred).astype(float)
    brier = np.mean((confidence - correct) ** 2)

    return f1 * (1 - brier)



print("Loading")
df = pd.read_pickle("../../../data/df_train.pkl")
df_train, df_val = train_test_split(df, test_size=0.1, random_state=42, stratify=df["industry"])
df_train["business_description_embedding"] = df_train["business_description_embedding"].apply(ast.literal_eval)
df_val["business_description_embedding"] = df_val["business_description_embedding"].apply(ast.literal_eval)

X_train = np.array(df_train["business_description_embedding"].tolist())
X_val = np.array(df_val["business_description_embedding"].tolist())

label_encoder = LabelEncoder().fit(df_train.industry)

y_train = label_encoder.transform(df_train.industry)
y_val = label_encoder.transform(df_val.industry)



def remove_duplicates(df):
    # Count exact duplicates
    exact_duplicates = df.business_description_embedding.duplicated().sum()
    print(f"Number of exact duplicate embeddings: {exact_duplicates}")

    df = df.drop_duplicates(subset=["business_description_embedding"])
    return df
print("removing duplicates")
df_train = remove_duplicates(df_train)

# Create embeddings and labels for this reduced dataset
X_train = np.array(df_train["business_description_embedding"].tolist())
y_train = label_encoder.transform(df_train.industry)

# val set
df_val = remove_duplicates(df_val)
X_val = np.array(df_val["business_description_embedding"].tolist())
y_val = label_encoder.transform(df_val.industry)


from sklearn.ensemble import BaggingClassifier
from sklearn.tree import DecisionTreeClassifier
print("training classifier")
bagging_clf = BaggingClassifier(
    estimator=DecisionTreeClassifier(),
    random_state=42,
    n_jobs=-1
)
bagging_clf.fit(X_train, y_train)

y_pred = bagging_clf.predict(X_val)
confidence = bagging_clf.predict_proba(X_val)[:, 1]
print(f"confidences: \n {confidence[:2}")



print(f"Score with Bagging: {confidence_score(y_val, y_pred, confidence):.4f}")
score_model(y_val, y_pred, print_results=True)
