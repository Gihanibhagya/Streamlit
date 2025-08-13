import streamlit as st
import pandas as pd
import pickle
import numpy as np
import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.ensemble import RandomForestClassifier

# ------------------ App Config ------------------
st.set_page_config(page_title="Academic Stress Predictor", layout="wide")
MODEL_FILE = "model.pkl"

# ------------------ Helper Functions ------------------
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

def save_model(model_data):
    with open(MODEL_FILE, "wb") as f:
        pickle.dump(model_data, f)

def load_model():
    if os.path.exists(MODEL_FILE):
        with open(MODEL_FILE, "rb") as f:
            return pickle.load(f)
    return None

# ------------------ Sidebar Navigation ------------------
st.sidebar.title("Navigation")
page = st.sidebar.radio("Go to", ["Home", "Data Exploration", "Train Model", "Model Prediction", "Model Performance"])

# ------------------ Home Page ------------------
if page == "Home":
    st.title("🎓 Academic Stress Predictor")
    st.image("https://images.unsplash.com/photo-1503676260728-1c00da094a0b", use_container_width=True)
    st.markdown("""
    Welcome to the **Academic Stress Predictor**!  
    This tool allows you to:
    - Upload your dataset
    - Train a machine learning model
    - Predict academic stress levels
    - View model performance metrics
    - Explore your data visually  
    Use the sidebar to navigate through the app.
    """)

# ------------------ Data Exploration Page ------------------
elif page == "Data Exploration":
    st.header("📊 Data Exploration")
    uploaded_file = st.file_uploader("Upload CSV Dataset for Exploration", type=["csv"])
    if uploaded_file:
        df = load_csv(uploaded_file)
        st.subheader("Dataset Preview")
        st.write(df.head())

        st.subheader("Basic Statistics")
        st.write(df.describe(include="all"))

        st.subheader("Missing Values")
        st.write(df.isnull().sum())

        # Numeric Distribution
        numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if numeric_cols:
            st.subheader("Numeric Column Distributions")
            for col in numeric_cols:
                fig, ax = plt.subplots()
                sns.histplot(df[col], kde=True, ax=ax)
                ax.set_title(f"Distribution of {col}")
                st.pyplot(fig)

        # Categorical Counts
        cat_cols = df.select_dtypes(include=['object', 'category']).columns.tolist()
        if cat_cols:
            st.subheader("Categorical Column Counts")
            for col in cat_cols:
                fig, ax = plt.subplots()
                df[col].value_counts().plot(kind='bar', ax=ax)
                ax.set_title(f"Value Counts of {col}")
                st.pyplot(fig)

# ------------------ Train Model Page ------------------
elif page == "Train Model":
    st.header("Train Model")
    uploaded_file = st.file_uploader("Upload CSV Dataset", type=["csv"])
    if uploaded_file:
        df = load_csv(uploaded_file)
        st.write("Preview of dataset:", df.head())

        # Drop Timestamp column if exists
        if "Timestamp" in df.columns:
            df = df.drop(columns=["Timestamp"])
            st.info("Dropped 'Timestamp' column for training.")

        target_col = st.selectbox("Select Target Column", df.columns)
        if st.button("Train Model"):
            try:
                X = df.drop(columns=[target_col])
                y = df[target_col]

                # Encode categorical variables
                encoders_dict = {}
                for col in X.select_dtypes(include=['object', 'category']).columns:
                    le = LabelEncoder()
                    X[col] = le.fit_transform(X[col])
                    encoders_dict[col] = le

                # Encode target
                target_encoder = LabelEncoder()
                y = target_encoder.fit_transform(y)

                # Train/test split
                X_train, X_test, y_train, y_test = train_test_split(
                    X, y, test_size=0.2, random_state=42
                )

                # Scale features
                scaler = StandardScaler()
                X_train = scaler.fit_transform(X_train)
                X_test = scaler.transform(X_test)

                # Train RandomForest
                model = RandomForestClassifier(random_state=42)
                model.fit(X_train, y_train)

                # Predictions
                y_pred = model.predict(X_test)

                # Metrics
                accuracy = accuracy_score(y_test, y_pred)
                class_report = classification_report(y_test, y_pred)
                conf_matrix = confusion_matrix(y_test, y_pred)

                # Save all to pickle
                model_data = {
                    "model": model,
                    "scaler": scaler,
                    "encoders": encoders_dict,
                    "target_encoder": target_encoder,
                    "feature_names": X.columns.tolist(),
                    "metrics": {
                        "accuracy": accuracy,
                        "classification_report": class_report,
                        "confusion_matrix": conf_matrix.tolist()
                    }
                }
                save_model(model_data)
                st.success(f"Model trained successfully! Accuracy: {accuracy:.2%}")
            except Exception as e:
                st.error(f"Training failed: {e}")

# ------------------ Model Prediction Page ------------------
elif page == "Model Prediction":
    st.header("Model Prediction")
    model_data = load_model()
    if not model_data:
        st.error("No model found. Please train the model first.")
    else:
        user_input = []
        for col in model_data["feature_names"]:
            if col in model_data["encoders"]:
                options = list(model_data["encoders"][col].classes_)
                val = st.selectbox(f"Select {col}", options)
                user_input.append(model_data["encoders"][col].transform([val])[0])
            else:
                val = st.number_input(f"Enter {col}", value=0.0)
                user_input.append(val)

        if st.button("Predict"):
            try:
                input_df = pd.DataFrame([user_input], columns=model_data["feature_names"])
                scaled_input = model_data["scaler"].transform(input_df)
                pred = model_data["model"].predict(scaled_input)[0]
                pred_label = model_data["target_encoder"].inverse_transform([pred])[0]
                st.success(f"Prediction: {pred_label}")
            except Exception as e:
                st.error(f"Prediction failed: {e}")

# ------------------ Model Performance Page ------------------
elif page == "Model Performance":
    st.header("Model Performance")
    model_data = load_model()
    if not model_data:
        st.error("No saved metrics found. Please train the model first.")
    else:
        metrics = model_data.get("metrics", None)
        if not metrics:
            st.error("No saved metrics found in model file.")
        else:
            st.write(f"**Accuracy:** {metrics['accuracy']:.2%}")
            st.text("Classification Report:")
            st.text(metrics["classification_report"])

            # Confusion Matrix Heatmap
            conf_matrix = np.array(metrics["confusion_matrix"])
            fig, ax = plt.subplots()
            sns.heatmap(conf_matrix, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_xlabel("Predicted")
            ax.set_ylabel("Actual")
            st.pyplot(fig)
