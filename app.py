import os
import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
import streamlit as st
from sklearn.metrics import (accuracy_score, classification_report,
                             confusion_matrix, f1_score, matthews_corrcoef,
                             precision_score, recall_score, roc_auc_score,
                             roc_curve)


TARGET = "Survived"

RAW_COLUMNS = ["PassengerId", "Survived", "Pclass", "Name", "Sex", "Age", "SibSp",
               "Parch", "Ticket", "Fare", "Cabin", "Embarked"]

NUMERIC_FEATURES = ["Pclass", "Age", "SibSp", "Parch", "Fare", "FamilySize",
                    "IsAlone", "HasCabin", "FarePerPerson"]

CATEGORICAL_FEATURES = ["Sex", "Embarked", "Title", "Deck"]

ALL_FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES  # 13 model input features

_RARE_TITLES = {"Lady", "Countess", "Capt", "Col", "Don", "Dr", "Major", "Rev",
                "Sir", "Jonkheer", "Dona"}


def _extract_title(name: str) -> str:
    title = name.split(",")[-1].split(".")[0].strip()
    if title in ("Mlle", "Ms"):
        return "Miss"
    if title == "Mme":
        return "Mrs"
    if title in _RARE_TITLES:
        return "Rare"
    return title


def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    # Turn raw columns into model features.
    d = df.copy()
    d["Title"] = d["Name"].astype(str).apply(_extract_title)
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    d["HasCabin"] = d["Cabin"].notna().astype(int)
    d["Deck"] = d["Cabin"].astype(str).str[0].where(d["Cabin"].notna(), "U")
    d["FarePerPerson"] = d["Fare"] / d["FamilySize"]
    return d[ALL_FEATURES]


st.set_page_config(page_title="Titanic Survival - Model Lab",
                   layout="wide")

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "KNN": "knn.joblib",
    "Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "support_vector_machine.joblib",
}
REQUIRED_COLS = [c for c in RAW_COLUMNS if c not in ("PassengerId",)]  # PassengerId optional
BASE_DIR = os.path.dirname(os.path.abspath(__file__))


@st.cache_resource
def load_model(name: str):
    return joblib.load(os.path.join(BASE_DIR, "model", MODEL_FILES[name]))


def compute_metrics(y_true, y_pred, y_proba) -> dict:
    return {
        "Accuracy": accuracy_score(y_true, y_pred),
        "AUC": roc_auc_score(y_true, y_proba),
        "Precision": precision_score(y_true, y_pred, zero_division=0),
        "Recall": recall_score(y_true, y_pred, zero_division=0),
        "F1": f1_score(y_true, y_pred, zero_division=0),
        "MCC": matthews_corrcoef(y_true, y_pred),
    }


# ---------- SIDEBAR -----------
st.sidebar.title(" Titanic Model Lab")
st.sidebar.caption("ML Assignment 2 - M.Tech (AIML), BITS Pilani")

uploaded = st.sidebar.file_uploader("1. Upload test CSV", type=["csv"],
                                    help="Must contain the raw Titanic columns including 'Survived'.")

model_name = st.sidebar.selectbox("2. Choose a model", list(MODEL_FILES.keys()))
st.sidebar.markdown("---")

# ---------- MAIN ----------
st.title("Titanic Survival Prediction - Classifier Evaluation")
st.write("Upload a test file, pick a model, and inspect how it performs. "
         "Target: **Survived** (1 = survived, 0 = did not survive).")

if uploaded is not None:
    df = pd.read_csv(uploaded)
else:
    st.warning("Upload a CSV to test.")
    st.stop()

missing = [c for c in REQUIRED_COLS if c not in df.columns]
if missing:
    st.error(f"Uploaded file is missing required columns: {missing}")
    st.stop()
if df[TARGET].isna().any():
    st.error("The 'Survived' column contains missing values.")
    st.stop()

y_true = df[TARGET].astype(int)
X = engineer_features(df.drop(columns=[TARGET]))

with st.expander(f"Preview of test data ({df.shape[0]} rows x {df.shape[1]} columns)"):
    st.dataframe(df.head(20), use_container_width=True)

# ---------- SELECT MODEL ----------
pipe = load_model(model_name)
y_pred = pipe.predict(X)
y_proba = pipe.predict_proba(X)[:, 1]
m = compute_metrics(y_true, y_pred, y_proba)

st.subheader(f"Results - {model_name}")
cols = st.columns(6)
for col, (k, v) in zip(cols, m.items()):
    col.metric(k, f"{v:.4f}")

left, right = st.columns(2)
with left:
    st.markdown("**Confusion matrix**")
    cm = confusion_matrix(y_true, y_pred, labels=[0, 1])
    fig, ax = plt.subplots(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=["Died", "Survived"], yticklabels=["Died", "Survived"])
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Actual")
    st.pyplot(fig)
with right:
    st.markdown("**ROC curve**")
    fpr, tpr, _ = roc_curve(y_true, y_proba)
    fig2, ax2 = plt.subplots(figsize=(4.2, 3.6))
    ax2.plot(fpr, tpr, label=f"AUC = {m['AUC']:.3f}")
    ax2.plot([0, 1], [0, 1], "--", color="grey")
    ax2.set_xlabel("False positive rate")
    ax2.set_ylabel("True positive rate")
    ax2.legend(loc="lower right")
    st.pyplot(fig2)

st.markdown("**Classification report**")
report = classification_report(y_true, y_pred, target_names=["Died", "Survived"],
                               output_dict=True, zero_division=0)
st.dataframe(pd.DataFrame(report).T.round(3), use_container_width=True)

# ---------- MODEL COMPARISON ----------
st.subheader("Comparison of all five models on this test data")
rows = []
for name in MODEL_FILES:
    p = load_model(name)
    rows.append({"ML Model Name": name,
                 **compute_metrics(y_true, p.predict(X), p.predict_proba(X)[:, 1])})
cmp_df = pd.DataFrame(rows).set_index("ML Model Name")
st.dataframe(cmp_df.style.format("{:.4f}").highlight_max(axis=0, color="#c6efce"),
             use_container_width=True)
best = cmp_df["MCC"].idxmax()
st.success(f"Best model on this data by MCC: **{best}** (MCC = {cmp_df.loc[best, 'MCC']:.4f})")
st.bar_chart(cmp_df[["Accuracy", "F1", "MCC"]])