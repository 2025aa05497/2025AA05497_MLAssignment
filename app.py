import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB

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
# Load Training Data (Cached)
# --------------------------------------------------
@st.cache_resource
def load_data():
    data = pd.read_csv("bank-full.csv", sep=";")
    data["target"] = data["y"].map({"yes": 1, "no": 0})
    data.drop("y", axis=1, inplace=True)

    encoded = pd.get_dummies(data, drop_first=True)
    X = encoded.drop("target", axis=1)
    y = encoded["target"]

    return X, y

X_train, y_train = load_data()

# --------------------------------------------------
# Train ALL Models Once (Instant Switching)
# --------------------------------------------------
@st.cache_resource
def train_all_models():
    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000),
        "Decision Tree": DecisionTreeClassifier(),
        "Random Forest": RandomForestClassifier(n_estimators=50),
        "KNN": KNeighborsClassifier(),
        "Naive Bayes": GaussianNB()
    }

    for model in models.values():
        model.fit(X_train, y_train)

    return models

models = train_all_models()

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

    # Encode test data
    test_encoded = pd.get_dummies(test_features, drop_first=True)

    # Align columns
    test_encoded = test_encoded.reindex(
        columns=X_train.columns,
        fill_value=0
    )

    # Predict
    y_pred = model.predict(test_encoded)

    st.success("Predictions generated successfully!")

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    if y_true is not None:

        if hasattr(model, "predict_proba"):
            y_prob = model.predict_proba(test_encoded)[:, 1]
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
