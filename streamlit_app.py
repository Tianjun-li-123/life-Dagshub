
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import streamlit as st
import io

from sklearn.impute import SimpleImputer, KNNImputer
# MLflow and DagsHub initialization
import mlflow
import mlflow.sklearn


import os
import dagshub
# if os.getenv("DAGSHUB_USERNAME") and os.getenv("DAGSHUB_TOKEN"):
#     import dagshub
#     dagshub.init(
#         repo_owner=os.getenv("DAGSHUB_USERNAME"),   
#         repo_name="life-exp-app",                   
#         mlflow=True                                 
#     )
dagshub.init(repo_owner='Tianjun-li-123', repo_name='DS4E-LIFE-EXP', mlflow=True)

import shap


from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor
from xgboost import XGBRegressor
from sklearn import metrics


def clean_missing(df: pd.DataFrame, numeric_strategy="median"):
    """
    Return a copy with missing values handled.

    numeric_strategy = "median" | "mean" | "knn"
    """
    df_clean = df.copy()

    num_cols = df_clean.select_dtypes(include="number").columns
    cat_cols = df_clean.select_dtypes(exclude="number").columns

    if numeric_strategy == "knn":
        knn = KNNImputer(n_neighbors=5)
        df_clean[num_cols] = knn.fit_transform(df_clean[num_cols])
    else:
        imp = SimpleImputer(strategy=numeric_strategy)
        df_clean[num_cols] = imp.fit_transform(df_clean[num_cols])

    if len(cat_cols):
        cat_imp = SimpleImputer(strategy="most_frequent")
        df_clean[cat_cols] = cat_imp.fit_transform(df_clean[cat_cols])

    return df_clean

st.set_page_config(
    page_title="Life Expectancy 🫀",
    layout="centered",
    page_icon="🫀",
)

st.sidebar.title("Life Expectancy Dashboard 🫀")
page = st.sidebar.selectbox(
    "Select Page",
    ["Introduction 📘", "Visualization 📊", "Prediction 🤖", "Explainability 🔍"]
)

st.image("life.jpg")
st.write("   ")
st.write("   ")
st.write("   ")

df = pd.read_csv("Life Expectancy Data.csv")

## Load Dataset
if page == "Introduction 📘":
    st.subheader("01 Introduction 📘")

    st.markdown("##### Data Preview")
    rows = st.slider("Select a number of rows to display", 5, 20, 5)
    st.dataframe(df.head(rows))

    st.markdown("##### 📝 DataFrame Info")
    with st.expander("Show data info output ⇣"):
        buffer = io.StringIO()
        df.info(buf=buffer)
        info_str = buffer.getvalue()
        st.code(info_str)

    st.markdown("##### Missing values")
    missing = df.isnull().sum()
    st.write(missing)

    if missing.sum() == 0:
        st.success("✅ No missing values found")
    else:
        st.warning("⚠️ Your raw data have missing values")

    st.markdown("##### 🧹 Handle Missing Values")
    methods = {
        "Drop rows with any NA": "drop",
        "Fill numeric with median": "median",
        "Fill numeric with mean": "mean",
        "KNN imputation (k=5)": "knn",
    }
    choice = st.selectbox("Choose a strategy", list(methods.keys()), key="na_strategy")

    if st.button("Apply strategy"):
        if methods[choice] == "drop":
            df = df.dropna().reset_index(drop=True)
        else:
            df = clean_missing(df, numeric_strategy=methods[choice])

        st.success("Missing-value handling applied 🎉")
        st.write("Remaining NA counts:")
        st.write(df.isna().sum())

        csv = df.to_csv(index=False).encode()
        st.download_button(
            "⬇️ Download cleaned CSV",
            csv,
            file_name="life_expectancy_clean.csv",
            mime="text/csv",
        )

    st.markdown("##### 📈 Summary Statistics")
    if st.button("Show Describe Table"):
        st.dataframe(df.describe())





elif page == "Prediction 🤖":
    st.subheader("04 Prediction with MLflow Tracking 🤖")

    # Data preprocessing
    df2 = df.dropna().copy()
    le = LabelEncoder()
    df2["Status"] = le.fit_transform(df2["Status"])
    df2 = df2.drop(columns=["Country"])  # Drop 'Country' string column

    # Feature/Target selection
    list_var = df2.columns.tolist()
    features_selection = st.sidebar.multiselect(
        "Select Features (X)",
        list_var,
        default=[col for col in list_var if col != "Life expectancy"],
    )
    target_selection = st.sidebar.selectbox(
        "Select Target Variable (Y)", list_var, index=list_var.index("Life expectancy ")
    )

    # Model choice
    model_name = st.sidebar.selectbox(
        "Choose Model",
        ["Linear Regression", "Decision Tree", "Random Forest", "XGBoost"],
    )

    # Hyperparameters
    params = {}
    if model_name == "Decision Tree":
        params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5)
    elif model_name == "Random Forest":
        params["n_estimators"] = st.sidebar.slider("Number of Estimators", 10, 500, 100)
        params["max_depth"] = st.sidebar.slider("Max Depth", 1, 20, 5)
    elif model_name == "XGBoost":
        params["n_estimators"] = st.sidebar.slider("Number of Estimators", 10, 500, 100)
        params["learning_rate"] = st.sidebar.slider(
            "Learning Rate", 0.01, 0.5, 0.1, step=0.01
        )

    selected_metrics = st.sidebar.multiselect(
        "Metrics to display",
        ["Mean Squared Error (MSE)", "Mean Absolute Error (MAE)", "R² Score"],
        default=["Mean Absolute Error (MAE)"],
    )

    # Prepare data
    X = df2[features_selection]
    y = df2[target_selection]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Instantiate model
    if model_name == "Linear Regression":
        model = LinearRegression()
    elif model_name == "Decision Tree":
        model = DecisionTreeRegressor(**params, random_state=42)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(**params, random_state=42)
    elif model_name == "XGBoost":
        model = XGBRegressor(
            objective="reg:squarederror", **params, random_state=42
        )

    # Train, predict and log with MLflow
    with mlflow.start_run(run_name=model_name):
        mlflow.log_param("model", model_name)
        for k, v in params.items():
            mlflow.log_param(k, v)

        model.fit(X_train, y_train)
        predictions = model.predict(X_test)

        # Log metrics
        mse = metrics.mean_squared_error(y_test, predictions)
        mae = metrics.mean_absolute_error(y_test, predictions)
        r2 = metrics.r2_score(y_test, predictions)
        mlflow.log_metric("mse", mse)
        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

    # Display metrics
    st.write(f"**MSE:** {mse:,.2f}")
    st.write(f"**MAE:** {mae:,.2f}")
    st.write(f"**R² Score:** {r2:.3f}")

    # Plot Actual vs Predicted
    fig, ax = plt.subplots()
    ax.scatter(y_test, predictions, alpha=0.5)
    ax.plot(
        [y_test.min(), y_test.max()],
        [y_test.min(), y_test.max()],
        "--r",
        linewidth=2,
    )
    ax.set_xlabel("Actual")
    ax.set_ylabel("Predicted")
    ax.set_title("Actual vs Predicted")
    st.pyplot(fig)


elif page == "Explainability 🔍":
    st.subheader("06 Explainability 🔍")

    # Load built-in California dataset for SHAP
    X_shap, y_shap = shap.datasets.california()

    # Train default XGBoost model for explainability
    model_exp = XGBRegressor(
        objective="reg:squarederror", n_estimators=100, random_state=42
    )
    model_exp.fit(X_shap, y_shap)

    explainer = shap.Explainer(model_exp)
    shap_values = explainer(X_shap)

    st.markdown("### SHAP Waterfall Plot for First Prediction")
    shap.plots.waterfall(shap_values[0], show=False)
    st.pyplot(plt.gcf())

    st.markdown("### SHAP Scatter Plot for 'Latitude'")
    shap.plots.scatter(shap_values[:, "Latitude"], color=shap_values, show=False)
    st.pyplot(plt.gcf())
  