"""
============================================================
AI FRAUD DETECTION - HIGH LEVEL MODEL TRAINING PIPELINE
============================================================

Pipeline:
1. Load and validate dataset
2. Clean data
3. Train / Validation / Test split
4. Feature-specific preprocessing
5. Handle class imbalance using SMOTE
6. Train multiple ML models
7. Evaluate using fraud-focused metrics
8. Select best model
9. Tune decision threshold
10. Save model + preprocessing pipeline
11. Generate visualizations and dashboard JSON
============================================================
"""

import os
import json
import warnings
import joblib
import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# ---------------------------------------------------------
# Machine Learning
# ---------------------------------------------------------
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    average_precision_score,
    precision_recall_curve
)

from imblearn.over_sampling import SMOTE
from xgboost import XGBClassifier

# ---------------------------------------------------------
# Visualization
# ---------------------------------------------------------
import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import seaborn as sns


# =========================================================
# CONFIGURATION
# =========================================================

DATA_PATH = "data/creditcard.csv"

MODEL_DIR = "model"
STATIC_DIR = "static"

RANDOM_STATE = 42

TEST_SIZE = 0.15
VALIDATION_SIZE = 0.15

SMOTE_RATIO = 0.30

os.makedirs(MODEL_DIR, exist_ok=True)
os.makedirs(STATIC_DIR, exist_ok=True)


print("=" * 70)
print("       AI FRAUD DETECTION MODEL TRAINING PIPELINE")
print("=" * 70)


# =========================================================
# STEP 1 — LOAD DATA
# =========================================================

print("\n[1/10] Loading dataset...")

if not os.path.exists(DATA_PATH):
    raise FileNotFoundError(
        f"Dataset not found at: {DATA_PATH}"
    )

df = pd.read_csv(DATA_PATH)

print(f"  Dataset shape : {df.shape}")
print(f"  Transactions  : {len(df):,}")


# =========================================================
# STEP 2 — DATA VALIDATION & CLEANING
# =========================================================

print("\n[2/10] Validating and cleaning data...")

# Check target column
if "Class" not in df.columns:
    raise ValueError(
        "Target column 'Class' was not found in dataset."
    )

# Remove duplicate rows
duplicate_count = df.duplicated().sum()

if duplicate_count > 0:
    print(f"  Removing duplicates: {duplicate_count:,}")
    df = df.drop_duplicates()

# Remove rows containing missing values
missing_count = df.isnull().sum().sum()

if missing_count > 0:
    print(f"  Missing values detected: {missing_count:,}")
    df = df.dropna()

# Make sure target is integer
df["Class"] = df["Class"].astype(int)

# Make sure target contains only 0 and 1
invalid_targets = ~df["Class"].isin([0, 1])

if invalid_targets.any():
    raise ValueError(
        "Target column contains values other than 0 and 1."
    )

print(f"  Clean dataset : {len(df):,} transactions")

fraud_count = int((df["Class"] == 1).sum())
legit_count = int((df["Class"] == 0).sum())

fraud_percentage = (fraud_count / len(df)) * 100

print(f"  Legit cases   : {legit_count:,}")
print(f"  Fraud cases   : {fraud_count:,}")
print(f"  Fraud rate    : {fraud_percentage:.4f}%")


# =========================================================
# STEP 3 — FEATURES & TARGET
# =========================================================

print("\n[3/10] Preparing features...")

X = df.drop(columns=["Class"])
y = df["Class"]

FEATURE_NAMES = list(X.columns)

print(f"  Total features: {len(FEATURE_NAMES)}")

print("\n  Features:")
print("  " + ", ".join(FEATURE_NAMES))


# =========================================================
# STEP 4 — TRAIN / VALIDATION / TEST SPLIT
# =========================================================

print("\n[4/10] Creating train / validation / test sets...")

# First split:
# 70% training
# 30% temporary
X_train, X_temp, y_train, y_temp = train_test_split(
    X,
    y,
    test_size=(TEST_SIZE + VALIDATION_SIZE),
    stratify=y,
    random_state=RANDOM_STATE
)

# Second split:
# Temporary → Validation + Test
validation_ratio = VALIDATION_SIZE / (
    TEST_SIZE + VALIDATION_SIZE
)

X_val, X_test, y_val, y_test = train_test_split(
    X_temp,
    y_temp,
    test_size=(TEST_SIZE / (TEST_SIZE + VALIDATION_SIZE)),
    stratify=y_temp,
    random_state=RANDOM_STATE
)

print(f"  Training   : {len(X_train):,}")
print(f"  Validation : {len(X_val):,}")
print(f"  Testing    : {len(X_test):,}")


# =========================================================
# STEP 5 — ADVANCED PREPROCESSING
# =========================================================

print("\n[5/10] Building preprocessing pipeline...")

# Time and Amount have different distributions.
#
# Time   → StandardScaler
# Amount → RobustScaler
#
# V1-V28 are PCA transformed features and are already
# approximately normalized, so we keep them unchanged.

scaled_features = ["Time", "Amount"]

passthrough_features = [
    feature
    for feature in FEATURE_NAMES
    if feature not in scaled_features
]

preprocessor = ColumnTransformer(
    transformers=[
        (
            "time_scaling",
            StandardScaler(),
            ["Time"]
        ),

        (
            "amount_scaling",
            RobustScaler(),
            ["Amount"]
        ),

        (
            "pca_features",
            "passthrough",
            passthrough_features
        )
    ],

    remainder="drop",

    verbose_feature_names_out=False
)

# IMPORTANT:
# Fit ONLY on training data.
#
# This prevents test/validation information from leaking
# into the model.

X_train_processed = preprocessor.fit_transform(X_train)

X_val_processed = preprocessor.transform(X_val)

X_test_processed = preprocessor.transform(X_test)

print("  Preprocessing fitted on training data only.")
print(f"  Processed training shape: {X_train_processed.shape}")
print(f"  Processed validation shape: {X_val_processed.shape}")
print(f"  Processed test shape: {X_test_processed.shape}")


# =========================================================
# STEP 6 — HANDLE CLASS IMBALANCE
# =========================================================

print("\n[6/10] Handling class imbalance using SMOTE...")

print("  Before SMOTE:")

before_distribution = pd.Series(
    y_train
).value_counts().sort_index()

print(
    f"    Legit : {before_distribution.get(0, 0):,}"
)

print(
    f"    Fraud : {before_distribution.get(1, 0):,}"
)

# We don't oversample validation or test data.
#
# SMOTE is applied ONLY to training data.

smote = SMOTE(
    sampling_strategy=SMOTE_RATIO,
    random_state=RANDOM_STATE,
    k_neighbors=5
)

X_train_balanced, y_train_balanced = smote.fit_resample(
    X_train_processed,
    y_train
)

after_distribution = pd.Series(
    y_train_balanced
).value_counts().sort_index()

print("\n  After SMOTE:")

print(
    f"    Legit : {after_distribution.get(0, 0):,}"
)

print(
    f"    Fraud : {after_distribution.get(1, 0):,}"
)


# =========================================================
# STEP 7 — TRAIN MULTIPLE MODELS
# =========================================================

print("\n[7/10] Training machine learning models...")

models = {

    "Random Forest": RandomForestClassifier(
        n_estimators=200,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "XGBoost": XGBClassifier(
        n_estimators=250,
        max_depth=6,
        learning_rate=0.08,
        subsample=0.85,
        colsample_bytree=0.85,
        objective="binary:logistic",
        eval_metric="logloss",
        random_state=RANDOM_STATE,
        n_jobs=-1
    ),

    "Logistic Regression": LogisticRegression(
        max_iter=2000,
        class_weight="balanced",
        random_state=RANDOM_STATE
    ),

    "Decision Tree": DecisionTreeClassifier(
        max_depth=12,
        min_samples_split=5,
        min_samples_leaf=2,
        class_weight="balanced",
        random_state=RANDOM_STATE
    )
}


trained_models = {}

for model_name, current_model in models.items():

    print(f"\n  Training: {model_name}")

    current_model.fit(
        X_train_balanced,
        y_train_balanced
    )

    trained_models[model_name] = current_model

    print(f"    ✓ {model_name} trained successfully")


# =========================================================
# STEP 8 — MODEL EVALUATION
# =========================================================

print("\n[8/10] Evaluating models...")

comparison_results = {}

for model_name, current_model in trained_models.items():

    # Validation probabilities
    val_probability = current_model.predict_proba(
        X_val_processed
    )[:, 1]

    val_prediction = (
        val_probability >= 0.50
    ).astype(int)

    accuracy = accuracy_score(
        y_val,
        val_prediction
    )

    precision = precision_score(
        y_val,
        val_prediction,
        zero_division=0
    )

    recall = recall_score(
        y_val,
        val_prediction,
        zero_division=0
    )

    f1 = f1_score(
        y_val,
        val_prediction,
        zero_division=0
    )

    roc_auc = roc_auc_score(
        y_val,
        val_probability
    )

    pr_auc = average_precision_score(
        y_val,
        val_probability
    )

    comparison_results[model_name] = {

        "accuracy": round(
            accuracy * 100,
            2
        ),

        "precision": round(
            precision * 100,
            2
        ),

        "recall": round(
            recall * 100,
            2
        ),

        "f1": round(
            f1 * 100,
            2
        ),

        "roc_auc": round(
            roc_auc * 100,
            2
        ),

        "pr_auc": round(
            pr_auc * 100,
            2
        )
    }

    print(
        f"  {model_name:<22}"
        f" F1={f1:.4f} | "
        f"ROC-AUC={roc_auc:.4f} | "
        f"PR-AUC={pr_auc:.4f}"
    )


# =========================================================
# STEP 9 — SELECT BEST MODEL
# =========================================================

print("\n[9/10] Selecting best model...")

# For fraud detection, PR-AUC is particularly useful
# because the dataset is highly imbalanced.

best_model_name = max(
    comparison_results,
    key=lambda name: comparison_results[name]["pr_auc"]
)

best_model = trained_models[
    best_model_name
]

print(
    f"\n  BEST MODEL: {best_model_name}"
)

print(
    f"  PR-AUC: "
    f"{comparison_results[best_model_name]['pr_auc']}%"
)


# =========================================================
# THRESHOLD OPTIMIZATION
# =========================================================

print("\n  Optimizing fraud decision threshold...")

validation_probability = best_model.predict_proba(
    X_val_processed
)[:, 1]

precision_curve, recall_curve, thresholds = precision_recall_curve(
    y_val,
    validation_probability
)

# Calculate F1 for each threshold
f1_scores = (
    2 *
    precision_curve *
    recall_curve /
    (
        precision_curve +
        recall_curve +
        1e-10
    )
)

best_threshold_index = np.argmax(
    f1_scores
)

if best_threshold_index >= len(thresholds):
    best_threshold = 0.50
else:
    best_threshold = thresholds[
        best_threshold_index
    ]

print(
    f"  Optimal threshold: "
    f"{best_threshold:.4f}"
)


# =========================================================
# FINAL TEST EVALUATION
# =========================================================

print("\n  Final evaluation on untouched TEST data...")

test_probability = best_model.predict_proba(
    X_test_processed
)[:, 1]

test_prediction = (
    test_probability >= best_threshold
).astype(int)


test_accuracy = accuracy_score(
    y_test,
    test_prediction
)

test_precision = precision_score(
    y_test,
    test_prediction,
    zero_division=0
)

test_recall = recall_score(
    y_test,
    test_prediction,
    zero_division=0
)

test_f1 = f1_score(
    y_test,
    test_prediction,
    zero_division=0
)

test_roc_auc = roc_auc_score(
    y_test,
    test_probability
)

test_pr_auc = average_precision_score(
    y_test,
    test_probability
)


print("\n" + "-" * 60)
print("FINAL TEST RESULTS")
print("-" * 60)

print(
    f"Accuracy  : {test_accuracy * 100:.2f}%"
)

print(
    f"Precision : {test_precision * 100:.2f}%"
)

print(
    f"Recall    : {test_recall * 100:.2f}%"
)

print(
    f"F1 Score  : {test_f1 * 100:.2f}%"
)

print(
    f"ROC-AUC   : {test_roc_auc * 100:.2f}%"
)

print(
    f"PR-AUC    : {test_pr_auc * 100:.2f}%"
)

print(
    f"Threshold : {best_threshold:.4f}"
)

print("\nClassification Report:\n")

print(
    classification_report(
        y_test,
        test_prediction,
        target_names=[
            "Legit",
            "Fraud"
        ],
        zero_division=0
    )
)


# =========================================================
# CONFUSION MATRIX
# =========================================================

print("\n  Generating confusion matrix...")

cm = confusion_matrix(
    y_test,
    test_prediction
)

plt.figure(
    figsize=(7, 5)
)

sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=[
        "Legit",
        "Fraud"
    ],
    yticklabels=[
        "Legit",
        "Fraud"
    ]
)

plt.title(
    f"Confusion Matrix - {best_model_name}"
)

plt.xlabel(
    "Predicted"
)

plt.ylabel(
    "Actual"
)

plt.tight_layout()

plt.savefig(
    os.path.join(
        STATIC_DIR,
        "confusion_matrix.png"
    ),
    dpi=200
)

plt.close()


# =========================================================
# FEATURE IMPORTANCE
# =========================================================

print("  Generating feature importance...")

if hasattr(
    best_model,
    "feature_importances_"
):

    importance_values = (
        best_model.feature_importances_
    )

    # Get transformed feature names
    transformed_features = (
        preprocessor.get_feature_names_out()
    )

    importance_df = pd.DataFrame({

        "Feature": transformed_features,

        "Importance": importance_values

    }).sort_values(
        "Importance",
        ascending=False
    ).head(15)

    plt.figure(
        figsize=(9, 6)
    )

    sns.barplot(
        data=importance_df,
        x="Importance",
        y="Feature"
    )

    plt.title(
        f"Top Fraud Detection Features - {best_model_name}"
    )

    plt.tight_layout()

    plt.savefig(
        os.path.join(
            STATIC_DIR,
            "feature_importance.png"
        ),
        dpi=200
    )

    plt.close()


# =========================================================
# MODEL COMPARISON JSON
# =========================================================

print("  Saving model comparison...")

with open(
    os.path.join(
        STATIC_DIR,
        "model_comparison.json"
    ),
    "w"
) as file:

    json.dump(
        comparison_results,
        file,
        indent=4
    )


# =========================================================
# MODEL COMPARISON CHART
# =========================================================

print("  Generating model comparison chart...")

model_names = list(
    comparison_results.keys()
)

pr_auc_scores = [
    comparison_results[name]["pr_auc"]
    for name in model_names
]

plt.figure(
    figsize=(10, 6)
)

bars = plt.bar(
    model_names,
    pr_auc_scores
)

plt.title(
    "Model Comparison - PR-AUC"
)

plt.ylabel(
    "PR-AUC (%)"
)

plt.ylim(
    0,
    max(pr_auc_scores) * 1.15
)

for bar, score in zip(
    bars,
    pr_auc_scores
):

    plt.text(
        bar.get_x()
        + bar.get_width() / 2,

        bar.get_height(),

        f"{score:.2f}%",

        ha="center",

        va="bottom"
    )

plt.tight_layout()

plt.savefig(
    os.path.join(
        STATIC_DIR,
        "model_comparison.png"
    ),
    dpi=200
)

plt.close()


# =========================================================
# SAVE PREPROCESSOR
# =========================================================

print("\n  Saving preprocessing pipeline...")

joblib.dump(
    preprocessor,
    os.path.join(
        MODEL_DIR,
        "preprocessor.pkl"
    )
)

print(
    "  ✓ model/preprocessor.pkl"
)


# =========================================================
# SAVE MODEL
# =========================================================

print("  Saving best model...")

joblib.dump(
    best_model,
    os.path.join(
        MODEL_DIR,
        "fraud_model.pkl"
    )
)

print(
    f"  ✓ model/fraud_model.pkl"
)


# =========================================================
# SAVE MODEL METADATA
# =========================================================

metadata = {

    "model_name": best_model_name,

    "features": FEATURE_NAMES,

    "threshold": float(
        best_threshold
    ),

    "training_samples": int(
        len(X_train)
    ),

    "validation_samples": int(
        len(X_val)
    ),

    "test_samples": int(
        len(X_test)
    ),

    "fraud_rate": float(
        fraud_percentage
    ),

    "metrics": {

        "accuracy": round(
            test_accuracy * 100,
            2
        ),

        "precision": round(
            test_precision * 100,
            2
        ),

        "recall": round(
            test_recall * 100,
            2
        ),

        "f1": round(
            test_f1 * 100,
            2
        ),

        "roc_auc": round(
            test_roc_auc * 100,
            2
        ),

        "pr_auc": round(
            test_pr_auc * 100,
            2
        )
    },

    "preprocessing": {

        "Time": "StandardScaler",

        "Amount": "RobustScaler",

        "V1-V28": "Passthrough - PCA transformed"
    },

    "imbalance_method": "SMOTE",

    "smote_ratio": SMOTE_RATIO,

    "random_state": RANDOM_STATE
}


with open(
    os.path.join(
        MODEL_DIR,
        "model_metadata.json"
    ),
    "w"
) as file:

    json.dump(
        metadata,
        file,
        indent=4
    )


# =========================================================
# SAVE TEST RESULTS
# =========================================================

test_results = pd.DataFrame({

    "actual": y_test.values,

    "fraud_probability": test_probability,

    "prediction": test_prediction

})

test_results.to_csv(
    os.path.join(
        STATIC_DIR,
        "test_predictions.csv"
    ),
    index=False
)


# =========================================================
# FINAL SUMMARY
# =========================================================

print("\n")
print("=" * 70)
print("              TRAINING COMPLETED SUCCESSFULLY")
print("=" * 70)

print(
    f"Best Model       : {best_model_name}"
)

print(
    f"ROC-AUC          : {test_roc_auc * 100:.2f}%"
)

print(
    f"PR-AUC           : {test_pr_auc * 100:.2f}%"
)

print(
    f"Precision        : {test_precision * 100:.2f}%"
)

print(
    f"Recall           : {test_recall * 100:.2f}%"
)

print(
    f"F1 Score         : {test_f1 * 100:.2f}%"
)

print(
    f"Fraud Threshold  : {best_threshold:.4f}"
)

print("\nGenerated files:")

print(
    "  ✓ model/fraud_model.pkl"
)

print(
    "  ✓ model/preprocessor.pkl"
)

print(
    "  ✓ model/model_metadata.json"
)

print(
    "  ✓ static/confusion_matrix.png"
)

print(
    "  ✓ static/feature_importance.png"
)

print(
    "  ✓ static/model_comparison.png"
)

print(
    "  ✓ static/model_comparison.json"
)

print(
    "  ✓ static/test_predictions.csv"
)

print("=" * 70)