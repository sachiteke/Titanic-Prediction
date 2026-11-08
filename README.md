# TITANIC SURVIVAL PREDICTION

## A. PROBLEM STATEMENT USED
Predict whether a passenger survived the Titanic disaster (`Survived`: 1 = survived, 0 = did not survive) from passenger details- Class, Sex, Age, Family Size, Fare, Cabin and Embarking Port. This is a **Binary Classification** problem. Five classifiers (Logistic Regression, Decision Tree, KNN, Gaussian Naive Bayes, SVM) are trained on the data Titanic-Dataset.csv, compared on six metrics, and deployed to Streamlit app that evaluates the saved models on  test_data.csv .

## B. DATASET DESCRIPTION
- **Source:** Titanic dataset (Kaggle - "Titanic Dataset: Titanic Survival Prediction Dataset").
- **Instances:** 891 passengers.
- **Raw columns:** PassengerId, Survived (target), Pclass, Name, Sex, Age, SibSp, Parch, Ticket, Fare, Cabin, Embarked.
- **Engineered model features (13):**  `Pclass`, `Age`, `SibSp`, `Parch`, `Fare`, `Sex`, `Embarked` + derived `Title` (from Name), `FamilySize`, `IsAlone`, `HasCabin`, `Deck` (from Cabin), `FarePerPerson`.
- **Dropped:** PassengerId, Name, Ticket, Cabin (after feature extraction).
- **Class balance:** ~38% survived / ~62% did not.
- **Split:** 80/20 --> 712 train / 179 test. The test set is in `test_data.csv`.
- **Preprocessing (inside each saved pipeline):** feature engineering -> imputation -> StandardScaler (numeric) / One-Hot (categorical) -> classifier. Because preprocessing is part of the saved pipeline, the app never retrains anything.

## c. GitHub repository link
`https://github.com/sachiteke/Titanic-Prediction`

Repository layout:
```
├── app.py                 # Streamlit app
├── features.py            # feature engineering shared by training and app
├── train_models.py        # trains, evaluates and saves all 5 models
├── requirements.txt
├── README.md
├── Titanic-Dataset.csv    # full dataset
├── test_data.csv          # held-out test data (upload this in the app)
└── model/                 # saved pipelines (*.joblib) + metrics.csv
```

## d. Models used

### Comparison table (test set, 179 passengers)
| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---|---|---|---|---|---|
| Logistic Regression | 0.8436 | 0.8727 | 0.8254 | 0.7536 | 0.7879 | 0.6661 |
| Decision Tree | 0.7709 | 0.8391 | 0.7414 | 0.6232 | 0.6772 | 0.5063 |
| KNN | 0.7765 | 0.8433 | 0.7164 | 0.6957 | 0.7059 | 0.5259 |
| Naive Bayes | 0.7654 | 0.7771 | 0.6484 | 0.8551 | 0.7375 | 0.5492 |
| Support Vector Machine | 0.8380 | 0.8499 | 0.8125 | 0.7536 | 0.7820 | 0.6545 |

Hyper-parameters: LogisticRegression(max_iter=1000); DecisionTree(max_depth=5, min_samples_leaf=5); KNN(k=7); GaussianNB(); SVC(kernel="rbf", C=1.0, probability=True).

### Observations
| ML Model Name | Observation about model performance |
|---|---|
| Logistic Regression | Best overall: highest accuracy (0.844), AUC (0.873) and MCC (0.666). The survival signal (sex, class, title) is close to linear after one-hot encoding, so a simple, well-regularised linear model fits it well and generalises cleanly. |
| Decision Tree | Weakest recall (0.623) and lowest MCC (0.506). Even with depth capped at 5 it fits quirks of the small training set and misses many true survivors. Easy to interpret but less stable. |
| KNN | Middling (accuracy 0.777, MCC 0.526). Distances over mixed one-hot and scaled numeric features are noisy and the result depends on k, so it trails the linear and kernel models. |
| Naive Bayes | Highest recall (0.855) but lowest precision (0.648) and lowest AUC (0.777). The independence assumption is violated by correlated features (Pclass/Fare/Deck, SibSp/Parch/FamilySize), so it over-predicts survival. Useful if missing a survivor is costlier than a false alarm. |
| Support Vector Machine | Very close second (accuracy 0.838, MCC 0.655). The RBF kernel captures some non-linear interactions, but on this small, mostly-linear problem it does not beat logistic regression. |
| **Overall winner for your dataset?** | **Logistic Regression** - best Accuracy, AUC, Precision, F1 and MCC, with SVM a close runner-up. Only Naive Bayes wins a single metric (Recall). With a 179-row test set, differences of ~1 point between LR and SVM are within noise. |

## Running locally
```bash
pip install -r requirements.txt
python train_models.py        # regenerates model/*.joblib, test_data.csv, metrics
streamlit run app.py
```

## Streamlit app features
- CSV upload of test data (or one-click use of the bundled `test_data.csv`); validates required columns
- Model selection dropdown (5 models)
- Accuracy, AUC, Precision, Recall, F1, MCC for the selected model
- Confusion matrix, ROC curve and classification report
- Side-by-side comparison of all five models on the uploaded data, plus downloadable predictions

**Live app:** `<PASTE YOUR STREAMLIT COMMUNITY CLOUD URL HERE>`
