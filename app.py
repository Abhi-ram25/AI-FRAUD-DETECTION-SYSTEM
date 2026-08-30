# ============================================
# FRAUD DETECTION - FLASK WEB SERVER
# app.py (WITH ALL 4 EXTENSIONS)
# ============================================

from flask import Flask, render_template, request, jsonify, make_response
import numpy as np
import pandas as pd
import joblib
import json
import csv
import io
import os
import uuid
from datetime import datetime

# ============================================================
# FLASK APPLICATION
# ============================================================

app = Flask(__name__)

# ============================================================
# CONFIGURATION
# ============================================================

MODEL_PATH = "model/fraud_model.pkl"
PREPROCESSOR_PATH = "model/preprocessor.pkl"
METADATA_PATH = "model/model_metadata.json"

DATASET_PATH = "data/creditcard.csv"

HISTORY_FILE = "static/prediction_history.json"
MODEL_COMPARISON_FILE = "static/model_comparison.json"

MAX_HISTORY = 100


# ============================================================
# FEATURE CONFIGURATION
# ============================================================

V_FEATURES = [
    f"V{i}"
    for i in range(1, 29)
]

FEATURE_NAMES = [
    "Time",
    *V_FEATURES,
    "Amount"
]

TARGET_NAME = "Class"

print("=" * 65)
print("        AI FRAUD DETECTION SYSTEM")
print("=" * 65)


# ============================================================
# LOAD MODEL
# ============================================================

print("\nLoading machine learning model...")

if not os.path.exists(MODEL_PATH):
    raise FileNotFoundError(
        f"Model not found: {MODEL_PATH}\n"
        "Run 'python train_model.py' first."
    )

model = joblib.load(MODEL_PATH)

print("✓ Fraud detection model loaded")


# ============================================================
# LOAD PREPROCESSOR
# ============================================================

print("Loading preprocessing pipeline...")

if not os.path.exists(PREPROCESSOR_PATH):
    raise FileNotFoundError(
        f"Preprocessor not found: {PREPROCESSOR_PATH}\n"
        "Run the upgraded train_model.py first."
    )

preprocessor = joblib.load(PREPROCESSOR_PATH)

print("✓ Preprocessor loaded")


# ============================================================
# LOAD MODEL METADATA
# ============================================================

if os.path.exists(METADATA_PATH):

    with open(
        METADATA_PATH,
        "r"
    ) as file:

        MODEL_METADATA = json.load(file)

    FRAUD_THRESHOLD = float(
        MODEL_METADATA.get(
            "threshold",
            0.50
        )
    )

    BEST_MODEL_NAME = MODEL_METADATA.get(
        "model_name",
        "Unknown"
    )

else:

    MODEL_METADATA = {}

    FRAUD_THRESHOLD = 0.50

    BEST_MODEL_NAME = "Unknown"

print(
    f"✓ Fraud threshold: {FRAUD_THRESHOLD:.4f}"
)

print(
    f"✓ Selected model: {BEST_MODEL_NAME}"
)


# ============================================================
# LOAD DATASET
# ============================================================

print("\nLoading transaction dataset...")

if os.path.exists(DATASET_PATH):

    dataset_df = pd.read_csv(
        DATASET_PATH
    )

    print(
        f"✓ Dataset loaded: "
        f"{len(dataset_df):,} transactions"
    )

else:

    dataset_df = None

    print(
        "⚠ Dataset not found."
    )


print("\nSystem ready!")
print("=" * 65)


# ============================================================
# HISTORY FUNCTIONS
# ============================================================

def load_history():

    if not os.path.exists(
        HISTORY_FILE
    ):
        return []

    try:

        with open(
            HISTORY_FILE,
            "r"
        ) as file:

            return json.load(file)

    except Exception:

        return []


def save_to_history(record):

    history = load_history()

    history.insert(
        0,
        record
    )

    history = history[
        :MAX_HISTORY
    ]

    os.makedirs(
        "static",
        exist_ok=True
    )

    with open(
        HISTORY_FILE,
        "w"
    ) as file:

        json.dump(
            history,
            file,
            indent=2
        )


# ============================================================
# FEATURE VALIDATION
# ============================================================

def validate_feature_columns(
    columns
):

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in columns
    ]

    return missing_features


# ============================================================
# TRANSACTION VALIDATION
# ============================================================

def validate_transaction(data):

    if not isinstance(
        data,
        dict
    ):

        return False, {
            "error":
                "Request body must contain JSON data."
        }

    missing_features = [
        feature
        for feature in FEATURE_NAMES
        if feature not in data
    ]

    if missing_features:

        return False, {

            "error":
                "Complete transaction data is required.",

            "missing_features":
                missing_features,

            "required_feature_count":
                len(FEATURE_NAMES)
        }

    # Validate numeric values

    invalid_features = []

    for feature in FEATURE_NAMES:

        try:

            value = float(
                data[feature]
            )

            if not np.isfinite(
                value
            ):

                invalid_features.append(
                    feature
                )

        except (
            ValueError,
            TypeError
        ):

            invalid_features.append(
                feature
            )

    if invalid_features:

        return False, {

            "error":
                "Some feature values are invalid.",

            "invalid_features":
                invalid_features
        }

    return True, None


# ============================================================
# PREPARE SINGLE TRANSACTION
# ============================================================

def prepare_transaction(
    data
):

    feature_values = {}

    for feature in FEATURE_NAMES:

        feature_values[feature] = float(
            data[feature]
        )

    features = pd.DataFrame(
        [feature_values],
        columns=FEATURE_NAMES
    )

    return features


# ============================================================
# RISK ENGINE
# ============================================================

def calculate_risk_score(
    fraud_probability
):

    score = fraud_probability * 100

    score = max(
        0,
        min(
            100,
            score
        )
    )

    return round(
        score,
        2
    )


def get_risk_level(
    risk_score
):

    if risk_score < 30:

        return "LOW"

    elif risk_score < 70:

        return "MEDIUM"

    return "HIGH"


def get_decision(
    risk_score
):

    if risk_score < 30:

        return "APPROVE"

    elif risk_score < 70:

        return "VERIFY"

    return "BLOCK"


# ============================================================
# SINGLE TRANSACTION PREDICTION
# ============================================================

def predict_transaction(
    data
):

    valid, error = validate_transaction(
        data
    )

    if not valid:

        return None, error

    try:

        # --------------------------------------------
        # Create raw feature DataFrame
        # --------------------------------------------

        features = prepare_transaction(
            data
        )

        # --------------------------------------------
        # Apply exact preprocessing used during training
        # --------------------------------------------

        processed_features = preprocessor.transform(
            features
        )

        # --------------------------------------------
        # ML Prediction
        # --------------------------------------------

        probability = model.predict_proba(
            processed_features
        )[0]

        fraud_probability = float(
            probability[1]
        )

        legit_probability = float(
            probability[0]
        )

        # --------------------------------------------
        # Optimized threshold
        # --------------------------------------------

        prediction = int(
            fraud_probability >= FRAUD_THRESHOLD
        )

        # --------------------------------------------
        # Risk engine
        # --------------------------------------------

        risk_score = calculate_risk_score(
            fraud_probability
        )

        risk_level = get_risk_level(
            risk_score
        )

        decision = get_decision(
            risk_score
        )

        return {

            "prediction":
                prediction,

            "label":
                (
                    "FRAUDULENT"
                    if prediction == 1
                    else "LEGITIMATE"
                ),

            "fraud_probability":
                round(
                    fraud_probability * 100,
                    2
                ),

            "legit_probability":
                round(
                    legit_probability * 100,
                    2
                ),

            "risk_score":
                risk_score,

            "risk_level":
                risk_level,

            "decision":
                decision,

            "threshold":
                round(
                    FRAUD_THRESHOLD * 100,
                    2
                ),

            "model":
                BEST_MODEL_NAME,

            "status":
                "success"

        }, None

    except Exception as e:

        print(
            f"Prediction error: {e}"
        )

        return None, {

            "error":
                "Prediction failed.",

            "details":
                str(e)
        }


# ============================================================
# HOME PAGE
# ============================================================

@app.route("/")
def home():

    return render_template(
        "index.html"
    )


# ============================================================
# HEALTH CHECK
# ============================================================

@app.route(
    "/health",
    methods=["GET"]
)
def health():

    return jsonify({

        "status":
            "healthy",

        "model_loaded":
            model is not None,

        "preprocessor_loaded":
            preprocessor is not None,

        "model":
            BEST_MODEL_NAME,

        "fraud_threshold":
            FRAUD_THRESHOLD,

        "features":
            len(FEATURE_NAMES)

    })


# ============================================================
# MAIN PREDICTION API
# ============================================================

@app.route(
    "/predict",
    methods=["POST"]
)
def predict():

    try:

        data = request.get_json(
            silent=True
        )

        result, error = predict_transaction(
            data
        )

        if error:

            return jsonify({

                "status":
                    "error",

                **error

            }), 400

        # --------------------------------------------
        # Transaction ID
        # --------------------------------------------

        transaction_id = (
            "TXN-"
            + uuid.uuid4().hex[:10].upper()
        )

        result[
            "transaction_id"
        ] = transaction_id

        # --------------------------------------------
        # Save history
        # --------------------------------------------

        amount = float(
            data["Amount"]
        )

        record = {

            "transaction_id":
                transaction_id,

            "timestamp":
                datetime.now().strftime(
                    "%Y-%m-%d %H:%M:%S"
                ),

            "amount":
                round(
                    amount,
                    2
                ),

            "prediction":
                result["label"],

            "fraud_prob":
                result[
                    "fraud_probability"
                ],

            "legit_prob":
                result[
                    "legit_probability"
                ],

            "risk_score":
                result[
                    "risk_score"
                ],

            "risk_level":
                result[
                    "risk_level"
                ],

            "decision":
                result[
                    "decision"
                ]
        }

        save_to_history(
            record
        )

        print(
            f"[{transaction_id}] "
            f"{result['label']} | "
            f"Risk={result['risk_score']} | "
            f"Decision={result['decision']}"
        )

        return jsonify(
            result
        )

    except Exception as e:

        print(
            f"Prediction API error: {e}"
        )

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# DEMO TRANSACTION PREDICTION
# ============================================================

@app.route(
    "/demo_predict",
    methods=["POST"]
)
def demo_predict():

    try:

        if dataset_df is None:

            return jsonify({

                "status":
                    "error",

                "message":
                    "Dataset is not available."

            }), 500

        data = request.get_json(
            silent=True
        )

        if not data:

            return jsonify({

                "status":
                    "error",

                "message":
                    "Transaction ID is required."

            }), 400

        transaction_id = int(
            data.get(
                "transaction_id",
                -1
            )
        )

        # Validate ID

        if (
            transaction_id < 0
            or
            transaction_id >= len(
                dataset_df
            )
        ):

            return jsonify({

                "status":
                    "error",

                "message":
                    "Transaction ID is out of range.",

                "valid_range":
                    [
                        0,
                        len(dataset_df) - 1
                    ]

            }), 400

        # --------------------------------------------
        # Get actual transaction
        # --------------------------------------------

        transaction = dataset_df.iloc[
            transaction_id
        ]

        feature_data = {

            feature:
                float(
                    transaction[feature]
                )

            for feature in FEATURE_NAMES
        }

        # --------------------------------------------
        # Prediction
        # --------------------------------------------

        result, error = predict_transaction(
            feature_data
        )

        if error:

            return jsonify({

                "status":
                    "error",

                **error

            }), 400

        # --------------------------------------------
        # Actual class is ONLY for demonstration
        # --------------------------------------------

        actual_class = int(
            transaction[TARGET_NAME]
        )

        result[
            "dataset_transaction_id"
        ] = transaction_id

        result[
            "actual_class"
        ] = actual_class

        result[
            "actual_label"
        ] = (
            "FRAUD"
            if actual_class == 1
            else "LEGIT"
        )

        result[
            "status"
        ] = "success"

        return jsonify(
            result
        )

    except Exception as e:

        print(
            f"Demo prediction error: {e}"
        )

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# GET RANDOM DEMO TRANSACTION
# ============================================================

@app.route(
    "/demo_random",
    methods=["GET"]
)
def demo_random():

    try:

        if dataset_df is None:

            return jsonify({

                "status":
                    "error",

                "message":
                    "Dataset unavailable."

            }), 500

        random_index = np.random.randint(
            0,
            len(dataset_df)
        )

        transaction = dataset_df.iloc[
            random_index
        ]

        return jsonify({

            "status":
                "success",

            "transaction_id":
                int(random_index),

            "amount":
                round(
                    float(
                        transaction["Amount"]
                    ),
                    2
                ),

            "actual_class":
                int(
                    transaction["Class"]
                )

        })

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# HISTORY API
# ============================================================

@app.route(
    "/history",
    methods=["GET"]
)
def history():

    return jsonify(
        load_history()
    )


# ============================================================
# CLEAR HISTORY
# ============================================================

@app.route(
    "/history/clear",
    methods=["POST"]
)
def clear_history():

    try:

        with open(
            HISTORY_FILE,
            "w"
        ) as file:

            json.dump(
                [],
                file
            )

        return jsonify({

            "status":
                "success",

            "message":
                "Prediction history cleared."

        })

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# EXPORT HISTORY AS CSV
# ============================================================

@app.route(
    "/export",
    methods=["GET"]
)
def export():

    try:

        history = load_history()

        output = io.StringIO()

        fieldnames = [

            "transaction_id",
            "timestamp",
            "amount",
            "prediction",
            "fraud_prob",
            "legit_prob",
            "risk_score",
            "risk_level",
            "decision"

        ]

        writer = csv.DictWriter(
            output,
            fieldnames=fieldnames
        )

        writer.writeheader()

        for record in history:

            writer.writerow({

                field:
                    record.get(
                        field,
                        ""
                    )

                for field in fieldnames

            })

        response = make_response(
            output.getvalue()
        )

        response.headers[
            "Content-Disposition"
        ] = (
            "attachment; "
            "filename=fraud_predictions.csv"
        )

        response.headers[
            "Content-Type"
        ] = "text/csv"

        return response

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# BATCH CSV PREDICTION
# ============================================================

@app.route(
    "/batch_predict",
    methods=["POST"]
)
def batch_predict():

    try:

        if "file" not in request.files:

            return jsonify({

                "status":
                    "error",

                "message":
                    "No CSV file uploaded."

            }), 400

        uploaded_file = request.files[
            "file"
        ]

        if uploaded_file.filename == "":

            return jsonify({

                "status":
                    "error",

                "message":
                    "No file selected."

            }), 400

        # --------------------------------------------
        # Read CSV
        # --------------------------------------------

        df_input = pd.read_csv(
            uploaded_file
        )

        # --------------------------------------------
        # Validate required columns
        # --------------------------------------------

        missing_features = (
            validate_feature_columns(
                df_input.columns
            )
        )

        if missing_features:

            return jsonify({

                "status":
                    "error",

                "message":
                    "CSV is missing required features.",

                "missing_features":
                    missing_features,

                "required_features":
                    FEATURE_NAMES

            }), 400

        # --------------------------------------------
        # Validate numeric data
        # --------------------------------------------

        X_input = df_input[
            FEATURE_NAMES
        ].copy()

        for feature in FEATURE_NAMES:

            X_input[feature] = pd.to_numeric(
                X_input[feature],
                errors="coerce"
            )

        invalid_rows = X_input.isnull().any(
            axis=1
        )

        invalid_count = int(
            invalid_rows.sum()
        )

        if invalid_count > 0:

            return jsonify({

                "status":
                    "error",

                "message":
                    "CSV contains invalid or missing numeric values.",

                "invalid_rows":
                    invalid_count

            }), 400

        # --------------------------------------------
        # Preprocess entire CSV at once
        # --------------------------------------------

        X_processed = preprocessor.transform(
            X_input
        )

        # --------------------------------------------
        # Batch prediction
        # --------------------------------------------

        probabilities = model.predict_proba(
            X_processed
        )[:, 1]

        predictions = (
            probabilities >= FRAUD_THRESHOLD
        ).astype(int)

        # --------------------------------------------
        # Create result DataFrame
        # --------------------------------------------

        results_df = pd.DataFrame({

            "transaction_id": [

                "BATCH-"
                + uuid.uuid4().hex[:8].upper()

                for _ in range(
                    len(df_input)
                )

            ],

            "amount": X_input[
                "Amount"
            ].round(2),

            "fraud_probability": (
                probabilities * 100
            ).round(2),

            "legit_probability": (
                (1 - probabilities) * 100
            ).round(2),

            "prediction": np.where(
                predictions == 1,
                "FRAUD",
                "LEGIT"
            )

        })

        # --------------------------------------------
        # Risk score
        # --------------------------------------------

        results_df[
            "risk_score"
        ] = (
            probabilities * 100
        ).round(2)

        # --------------------------------------------
        # Risk level
        # --------------------------------------------

        results_df[
            "risk_level"
        ] = results_df[
            "risk_score"
        ].apply(
            get_risk_level
        )

        # --------------------------------------------
        # Decision
        # --------------------------------------------

        results_df[
            "decision"
        ] = results_df[
            "risk_score"
        ].apply(
            get_decision
        )

        # --------------------------------------------
        # Statistics
        # --------------------------------------------

        fraud_count = int(
            (
                predictions == 1
            ).sum()
        )

        legit_count = (
            len(predictions)
            - fraud_count
        )

        high_risk_count = int(
            (
                results_df[
                    "risk_level"
                ] == "HIGH"
            ).sum()
        )

        medium_risk_count = int(
            (
                results_df[
                    "risk_level"
                ] == "MEDIUM"
            ).sum()
        )

        low_risk_count = int(
            (
                results_df[
                    "risk_level"
                ] == "LOW"
            ).sum()
        )

        # --------------------------------------------
        # Return results
        # --------------------------------------------

        return jsonify({

            "status":
                "success",

            "total":
                len(results_df),

            "fraud_count":
                fraud_count,

            "legit_count":
                legit_count,

            "high_risk_count":
                high_risk_count,

            "medium_risk_count":
                medium_risk_count,

            "low_risk_count":
                low_risk_count,

            "fraud_percentage":
                round(
                    fraud_count
                    / len(results_df)
                    * 100,
                    2
                )
                if len(results_df) > 0
                else 0,

            "results":
                results_df.to_dict(
                    orient="records"
                )

        })

    except Exception as e:

        print(
            f"Batch prediction error: {e}"
        )

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# MODEL COMPARISON API
# ============================================================

@app.route(
    "/model_comparison",
    methods=["GET"]
)
def model_comparison():

    try:

        if not os.path.exists(
            MODEL_COMPARISON_FILE
        ):

            return jsonify({})

        with open(
            MODEL_COMPARISON_FILE,
            "r"
        ) as file:

            data = json.load(
                file
            )

        return jsonify(
            data
        )

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# FEATURE INFORMATION API
# ============================================================

@app.route(
    "/features",
    methods=["GET"]
)
def features():

    return jsonify({

        "total_features":
            len(FEATURE_NAMES),

        "features":
            FEATURE_NAMES,

        "v_features":
            V_FEATURES,

        "target":
            TARGET_NAME,

        "description":
            "V1-V28 are anonymized PCA-transformed features."

    })


# ============================================================
# MODEL INFORMATION API
# ============================================================

@app.route(
    "/model_info",
    methods=["GET"]
)
def model_info():

    return jsonify({

        "model":
            BEST_MODEL_NAME,

        "fraud_threshold":
            round(
                FRAUD_THRESHOLD,
                4
            ),

        "fraud_threshold_percentage":
            round(
                FRAUD_THRESHOLD * 100,
                2
            ),

        "features":
            len(FEATURE_NAMES),

        "preprocessing":
            MODEL_METADATA.get(
                "preprocessing",
                {}
            ),

        "imbalance_method":
            MODEL_METADATA.get(
                "imbalance_method",
                "SMOTE"
            ),

        "metrics":
            MODEL_METADATA.get(
                "metrics",
                {}
            )

    })


# ============================================================
# DYNAMIC STATISTICS API
# ============================================================

@app.route(
    "/stats",
    methods=["GET"]
)
def stats():

    try:

        # --------------------------------------------
        # Dataset statistics
        # --------------------------------------------

        if dataset_df is not None:

            total_transactions = len(
                dataset_df
            )

            fraud_count = int(
                (
                    dataset_df[
                        "Class"
                    ] == 1
                ).sum()
            )

            legit_count = int(
                (
                    dataset_df[
                        "Class"
                    ] == 0
                ).sum()
            )

            fraud_percentage = round(
                fraud_count
                / total_transactions
                * 100,
                2
            )

            avg_fraud_amount = round(
                dataset_df[
                    dataset_df[
                        "Class"
                    ] == 1
                ]["Amount"].mean(),
                2
            )

            avg_legit_amount = round(
                dataset_df[
                    dataset_df[
                        "Class"
                    ] == 0
                ]["Amount"].mean(),
                2
            )

        else:

            total_transactions = 0
            fraud_count = 0
            legit_count = 0
            fraud_percentage = 0
            avg_fraud_amount = 0
            avg_legit_amount = 0

        # --------------------------------------------
        # Application history statistics
        # --------------------------------------------

        history = load_history()

        detected_fraud = sum(
            1
            for record in history
            if record.get(
                "prediction"
            ) == "FRAUDULENT"
            or
            record.get(
                "prediction"
            ) == "FRAUD"
        )

        blocked_count = sum(
            1
            for record in history
            if record.get(
                "decision"
            ) == "BLOCK"
        )

        return jsonify({

            "total_transactions":
                total_transactions,

            "fraud_count":
                fraud_count,

            "legit_count":
                legit_count,

            "fraud_percentage":
                fraud_percentage,

            "avg_fraud_amount":
                avg_fraud_amount,

            "avg_legit_amount":
                avg_legit_amount,

            "application_predictions":
                len(history),

            "application_fraud_detected":
                detected_fraud,

            "application_blocked":
                blocked_count,

            "model":
                BEST_MODEL_NAME,

            "threshold":
                round(
                    FRAUD_THRESHOLD * 100,
                    2
                )

        })

    except Exception as e:

        return jsonify({

            "status":
                "error",

            "message":
                str(e)

        }), 500


# ============================================================
# ERROR HANDLERS
# ============================================================

@app.errorhandler(404)
def page_not_found(error):

    return jsonify({

        "status":
            "error",

        "message":
            "Endpoint not found."

    }), 404


@app.errorhandler(500)
def internal_server_error(error):

    return jsonify({

        "status":
            "error",

        "message":
            "Internal server error."

    }), 500


# ============================================================
# APPLICATION START
# ============================================================

if __name__ == "__main__":

    app.run(
        host="127.0.0.1",
        port=5000,
        debug=True
    )