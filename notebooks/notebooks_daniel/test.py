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
import time


# Reload the preprocessed data
df_train = pd.read_pickle("../../data/processed/df_train_preprocessed.pkl")
df_val = pd.read_pickle("../../data/processed/df_val_preprocessed.pkl")

X_train = np.array(df_train["business_description_embedding"].tolist())
X_val = np.array(df_val["business_description_embedding"].tolist())

label_encoder = LabelEncoder().fit(df_train.industry)
y_train = label_encoder.transform(df_train.industry)
y_val = label_encoder.transform(df_val.industry)



from skopt import BayesSearchCV
from xgboost import XGBClassifier

xgb = XGBClassifier(eval_metric='logloss', 
                    device='cuda', 
                    tree_method='hist',
                    random_state=42,
                    learning_rate=0.3,
                    n_estimators=200, max_depth=6,
                    subsample= 0.2, colsample_bytree=0.2, 
                    verbosity=3,sampling_method='gradient_based')

time_start = time.time()
xgb.fit(X_train, y_train)
time_end = time.time()
print(f"Time taken: {time_end - time_start} seconds")
print(xgb.score(X_val, y_val))