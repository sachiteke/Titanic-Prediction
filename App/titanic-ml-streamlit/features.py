"""Shared feature-engineering + pipeline helpers (used by training script AND the Streamlit app)."""
import numpy as np
import pandas as pd

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
    """Turn raw Titanic columns into the 13 model features."""
    d = df.copy()
    d["Title"] = d["Name"].astype(str).apply(_extract_title)
    d["FamilySize"] = d["SibSp"] + d["Parch"] + 1
    d["IsAlone"] = (d["FamilySize"] == 1).astype(int)
    d["HasCabin"] = d["Cabin"].notna().astype(int)
    d["Deck"] = d["Cabin"].astype(str).str[0].where(d["Cabin"].notna(), "U")
    d["FarePerPerson"] = d["Fare"] / d["FamilySize"]
    return d[ALL_FEATURES]
