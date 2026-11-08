"""Train 5 classifiers on the Titanic dataset, evaluate, and save artefacts.

Run:  python train_models.py
Outputs: model/*.joblib, test_data.csv, model/metrics.csv
"""
import os
import joblib
import numpy as np
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, roc_auc_score, precision_score,
                             recall_score, f1_score, matthews_corrcoef)
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer, OneHotEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from features import (TARGET, NUMERIC_FEATURES, CATEGORICAL_FEATURES,
                      engineer_features)

SEED = 42
DATA_PATH = "Titanic-Dataset.csv"
MODEL_DIR = "model"
os.makedirs(MODEL_DIR, exist_ok=True)


def build_pipeline(clf):
    pre = ColumnTransformer([
        ("num", Pipeline([("imp", SimpleImputer(strategy="median")),
                          ("sc", StandardScaler())]), NUMERIC_FEATURES),
        ("cat", Pipeline([("imp", SimpleImputer(strategy="most_frequent")),
                          ("oh", OneHotEncoder(handle_unknown="ignore",
                                               sparse_output=False))]),
         CATEGORICAL_FEATURES),
    ])
    return Pipeline([("fe", FunctionTransformer(engineer_features)),
                     ("pre", pre), ("clf", clf)])


MODELS = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=SEED),
    "Decision Tree": DecisionTreeClassifier(max_depth=5, min_samples_leaf=5,
                                            random_state=SEED),
    "KNN": KNeighborsClassifier(n_neighbors=7),
    "Naive Bayes": GaussianNB(),
    "Support Vector Machine": SVC(kernel="rbf", C=1.0, probability=True,
                                  random_state=SEED),
}


def main():
    df = pd.read_csv(DATA_PATH)
    train_df, test_df = train_test_split(df, test_size=0.2, stratify=df[TARGET],
                                         random_state=SEED)
    # Raw-format test data (the Streamlit app applies the same feature engineering)
    test_df.to_csv("test_data.csv", index=False)

    X_train, y_train = train_df.drop(columns=[TARGET]), train_df[TARGET]
    X_test, y_test = test_df.drop(columns=[TARGET]), test_df[TARGET]

    rows = []
    for name, clf in MODELS.items():
        pipe = build_pipeline(clf).fit(X_train, y_train)
        pred = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        rows.append({
            "ML Model Name": name,
            "Accuracy": accuracy_score(y_test, pred),
            "AUC": roc_auc_score(y_test, proba),
            "Precision": precision_score(y_test, pred),
            "Recall": recall_score(y_test, pred),
            "F1": f1_score(y_test, pred),
            "MCC": matthews_corrcoef(y_test, pred),
        })
        fname = name.lower().replace(" ", "_") + ".joblib"
        joblib.dump(pipe, os.path.join(MODEL_DIR, fname))

    metrics = pd.DataFrame(rows).round(4)
    metrics.to_csv(os.path.join(MODEL_DIR, "metrics.csv"), index=False)
    print(metrics.to_string(index=False))


if __name__ == "__main__":
    main()
