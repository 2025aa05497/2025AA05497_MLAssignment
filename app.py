import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------------------------
# Page Config
# --------------------------------------------------
st.set_page_config(page_title="ML Assignment 2 - Classification", layout="wide")

st.title("📊 ML Assignment 2 – Classification")
st.write("Bank Marketing Dataset – Multiple ML Models Comparison")

# --------------------------------------------------
# Load Saved Models (NO TRAINING HERE)
# --------------------------------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("models/saved_models/logistic_regression.pkl"),
        "Decision Tree Classifier": joblib.load("models/saved_models/decision_tree_classifier.pkl"),
        "K-Nearest Neighbor Classifier": joblib.load("models/saved_models/knearest_neighbor_classifier.pkl"),
        "Naive Bayes Classifier": joblib.load("models/saved_models/naive_bayes_classifier.pkl"),
        "Ensemble Model - Random Forest": joblib.load("models/saved_models/ensemble_model__random_forest.pkl"),
        "Ensemble Model - XGBoost": joblib.load("models/saved_models/ensemble_model__xgboost.pkl"),
    }

    scaler = joblib.load("models/saved_models/scaler.pkl")

    return models, scaler


models, scaler = load_models()

# --------------------------------------------------
# Sidebar Controls
# --------------------------------------------------
st.sidebar.title("⚙️ Model Controls")

model_name = st.sidebar.selectbox(
    "Select Classification Model",
    list(models.keys())
)

uploaded_file = st.sidebar.file_uploader(
    "Upload Test CSV File",
    type=["csv"]
)

model = models[model_name]

# --------------------------------------------------
# Download Sample Test File
# --------------------------------------------------
st.subheader("⬇ Download Sample Test Dataset")

try:
    with open("test_data.csv", "rb") as f:
        st.download_button(
            "Download Test Data (CSV)",
            f,
            file_name="test_data.csv",
            mime="text/csv"
        )
except:
    st.info("Place 'test_data.csv' in the same folder to enable download.")

# --------------------------------------------------
# If User Uploads Test File
# --------------------------------------------------
if uploaded_file is not None:

    test_data = pd.read_csv(uploaded_file)

    y_true = None
    if "target" in test_data.columns:
        y_true = test_data["target"]
        test_features = test_data.drop("target", axis=1)
    else:
        test_features = test_data.copy()

    # Scale ONLY for Logistic & KNN
    if model_name in ["Logistic Regression", "K-Nearest Neighbor Classifier"]:
        test_features = scaler.transform(test_features)

    # Predict
    y_pred = model.predict(test_features)

    st.success("Predictions generated successfully!")

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    if y_true is not None:

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(test_features)[:, 1]
            auc = roc_auc_score(y_true, y_prob)
        else:
            auc = 0

        accuracy = accuracy_score(y_true, y_pred)
        precision = precision_score(y_true, y_pred, zero_division=0)
        recall = recall_score(y_true, y_pred, zero_division=0)
        f1 = f1_score(y_true, y_pred, zero_division=0)
        mcc = matthews_corrcoef(y_true, y_pred)

        st.subheader(f"📈 Evaluation Metrics – {model_name}")

        col1, col2 = st.columns(2)

        col1.metric("Accuracy", f"{accuracy:.4f}")
        col1.metric("Recall", f"{recall:.4f}")
        col1.metric("MCC", f"{mcc:.4f}")

        col2.metric("AUC Score", f"{auc:.4f}")
        col2.metric("F1 Score", f"{f1:.4f}")
        col2.metric("Precision", f"{precision:.4f}")

        # --------------------------------------------------
        # Confusion Matrix
        # --------------------------------------------------
        st.subheader("📊 Confusion Matrix")

        cm = confusion_matrix(y_true, y_pred)

        fig, ax = plt.subplots(figsize=(5, 4))
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=["No", "Yes"],
            yticklabels=["No", "Yes"],
            ax=ax
        )

        ax.set_xlabel("Predicted Label")
        ax.set_ylabel("True Label")

        st.pyplot(fig)

        # --------------------------------------------------
        # Classification Report
        # --------------------------------------------------
        st.subheader("📄 Classification Report")

        report = classification_report(y_true, y_pred, output_dict=True)
        report_df = pd.DataFrame(report).transpose()
        st.dataframe(report_df)

    else:
        st.warning("Upload test file with 'target' column to see evaluation metrics.")