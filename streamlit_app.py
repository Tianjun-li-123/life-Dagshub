import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
from datetime import datetime
import warnings
import os
import io
import requests
import sys
import mlflow
import mlflow.sklearn
import dagshub # Will be imported if MLFLOW_AVAILABLE and dagshub.init is called
import joblib # For saving/loading models locally for MLflow artifact workaround

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import shap # Will be imported if SHAP_AVAILABLE
import matplotlib.pyplot as plt
import seaborn as sns # Used in Visualization page

warnings.filterwarnings('ignore') # Suppress warnings globally

# --- Check if running with `streamlit run` ---
if 'streamlit' not in sys.modules or not hasattr(sys.modules['streamlit'], 'runtime'):
    print("Error: Please run this script with 'streamlit run streamlit_app.py' from your project root.")
    print("Example: cd /workspaces/life-Dagshub/ && streamlit run streamlit_app.py")
    sys.exit(1)

# --- Module Availability Checks (Global) ---
MLFLOW_AVAILABLE = False
try:
    import mlflow
    import mlflow.sklearn
    MLFLOW_AVAILABLE = True
except ImportError:
    st.warning("MLflow not installed. Experiment tracking features will be limited.")

PYCARET_AVAILABLE = False
try:
    from pycaret.regression import setup as reg_setup, compare_models as reg_compare, \
                                   finalize_model as reg_finalize, predict_model as reg_predict, \
                                   pull as reg_pull, save_model as reg_save_model, \
                                   create_model as reg_create, tune_model as reg_tune
    PYCARET_AVAILABLE = True
except ImportError:
    st.warning("PyCaret not installed. AutoML features will be limited.")
    # Define placeholder functions to prevent script crashes if PyCaret is missing
    reg_setup = reg_compare = reg_finalize = reg_predict = reg_pull = reg_save_model = reg_create = reg_tune = lambda *args, **kwargs: None


SHAP_AVAILABLE = False
try:
    import shap
    SHAP_AVAILABLE = True
    shap.initjs() # Initialize SHAP JS for interactive plots
except ImportError:
    st.warning("SHAP not installed. Model explainability features will be limited.")


# --- Utility Function: clean_missing (Previously data_preprocessing.py) ---
def clean_missing(df_input, numeric_strategy="median"):
    """
    Cleans the Life Expectancy DataFrame:
    - Converts 'Status' column to numerical (0: Developing, 1: Developed).
    - Drops 'Country' column.
    - Imputes missing numerical values using specified strategy (median or mean).
    - Imputes missing categorical/object values using mode.
    - Drops rows where the 'Life expectancy ' target is missing.
    """
    df_clean = df_input.copy()

    # Drop rows with missing target value first, as these cannot be used for training
    initial_rows = len(df_clean)
    if "Life expectancy " in df_clean.columns:
        df_clean.dropna(subset=["Life expectancy "], inplace=True)
        # st.info(f"Dropped {initial_rows - len(df_clean)} rows with missing target values.") # Handled in load_and_preprocess_data

    # Handle 'Status' column (binary encoding)
    if "Status" in df_clean.columns:
        df_clean["Status"] = df_clean["Status"].replace({"Developing": 0, "Developed": 1, "developing":0, "developed":1})
        df_clean["Status"] = pd.to_numeric(df_clean["Status"], errors='coerce') # Convert to numeric, errors as NaN

    # Drop 'Country' if present, as it's a unique identifier and not a feature
    if "Country" in df_clean.columns:
        df_clean = df_clean.drop(columns=["Country"])

    # Impute missing values for remaining columns
    for column in df_clean.columns:
        if df_clean[column].isnull().any():
            if pd.api.types.is_numeric_dtype(df_clean[column]):
                if numeric_strategy == "median":
                    df_clean[column].fillna(df_clean[column].median(), inplace=True)
                elif numeric_strategy == "mean":
                    df_clean[column].fillna(df_clean[column].mean(), inplace=True)
            elif df_clean[column].dtype == 'object' or pd.api.types.is_categorical_dtype(df_clean[column]):
                df_clean[column].fillna(df_clean[column].mode()[0], inplace=True)
    
    return df_clean

# --- Utility Function: train_classical_models (Previously model_training.py) ---
def train_classical_models(X_train, X_test, y_train, y_test, features):
    """
    Trains Linear Regression and Random Forest models with MLflow tracking.
    """
    if X_train is None or y_train is None or X_test is None or y_test is None or X_train.shape[0] == 0:
        st.error("Cannot train classical models: Missing or empty training/testing data.")
        return {}, {}
    
    models = {}
    metrics = {}

    if not MLFLOW_AVAILABLE:
        st.warning("MLflow not available. Skipping experiment tracking for classical models.")

    # Example input for MLflow signature
    input_example = pd.DataFrame(X_train[:1], columns=features) if len(X_train) > 0 else None

    # Linear Regression
    try:
        with mlflow.start_run(run_name="Linear_Regression"):
            lr = LinearRegression()
            lr.fit(X_train, y_train)
            y_pred = lr.predict(X_test)
            metrics_lr = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred)
            }
            if MLFLOW_AVAILABLE:
                mlflow.log_params({"model": "LinearRegression"})
                mlflow.log_metrics(metrics_lr)
                mlflow.sklearn.log_model(lr, artifact_path="linear_regression_model", input_example=input_example)
            
            models["Linear Regression"] = lr
            metrics["Linear Regression"] = metrics_lr
            st.success("✅ Linear Regression trained and metrics logged.")
    except Exception as e:
        st.error(f"❌ Error training Linear Regression: {str(e)}")

    # Random Forest with GridSearchCV
    try:
        with mlflow.start_run(run_name="Random_Forest"):
            param_grid = {
                "n_estimators": [50, 100],
                "max_depth": [None, 10, 20],
                "min_samples_split": [2, 5]
            }
            rf = RandomForestRegressor(random_state=42)
            grid_search = GridSearchCV(rf, param_grid, cv=3, scoring="neg_mean_squared_error", n_jobs=-1)
            grid_search.fit(X_train, y_train)
            best_rf = grid_search.best_estimator_
            y_pred = best_rf.predict(X_test)
            metrics_rf = {
                "MAE": mean_absolute_error(y_test, y_pred),
                "MSE": mean_squared_error(y_test, y_pred),
                "R2": r2_score(y_test, y_pred)
            }
            if MLFLOW_AVAILABLE:
                mlflow.log_params(grid_search.best_params_)
                mlflow.log_metrics(metrics_rf)
                mlflow.sklearn.log_model(best_rf, artifact_path="random_forest_model", input_example=input_example)
            
            models["Random Forest"] = best_rf
            metrics["Random Forest"] = metrics_rf
            st.success("✅ Random Forest trained and metrics logged.")
    except Exception as e:
        st.error(f"❌ Error training Random Forest: {str(e)}")

    return models, metrics

# --- Utility Function: shap_explain (Previously part of model_training.py) ---
def shap_explain(model, X_test, features):
    """
    Generates SHAP values for a given model and test set.
    """
    if X_test is None or model is None or features is None or X_test.shape[0] == 0:
        st.warning("Cannot generate SHAP plots: Missing model, test data, or features.")
        return None, None
    
    if not SHAP_AVAILABLE:
        st.error("SHAP library not available. Cannot generate explainability plots.")
        return None, None

    try:
        # Use a subset of X_test for explainer to avoid memory issues with large datasets
        subset_X_test = X_test[:min(len(X_test), 200)] 
        
        # Determine explainer type based on model
        if hasattr(model, "predict_proba"): # For classification
            explainer = shap.Explainer(model.predict_proba, subset_X_test)
        else: # For regression
            explainer = shap.Explainer(model.predict, subset_X_test)
            
        shap_values = explainer(subset_X_test)

        # If it's a multi-output model or multi-class, shap_values might be a list or 3D array
        if isinstance(shap_values, list): # For multi-output/multi-class shap_values
            shap_values = shap_values[0] # Take the first output if multiple
        elif len(shap_values.shape) == 3: # (samples, features, classes) or (samples, features, outputs)
            shap_values = shap_values[:, :, 0] # Take the first class/output

        # Ensure shap_values.data is available for plotting functions
        if not hasattr(shap_values, 'data') or shap_values.data is None:
             shap_values.data = subset_X_test
        if not hasattr(shap_values, 'feature_names') or shap_values.feature_names is None:
             shap_values.feature_names = features


        return shap_values, features
    except Exception as e:
        st.error(f"❌ An error occurred during SHAP explanation: {str(e)}. "
                 "This might be due to model type, data format, or SHAP version compatibility.")
        st.exception(e) # Display full traceback for debugging
        return None, None


# --- Streamlit Page Configuration ---
st.set_page_config(
    page_title="Life Expectancy Prediction 🫀",
    layout="wide",
    initial_sidebar_state="expanded",
    page_icon="🫀"
)

# --- Custom CSS Styling ---
st.markdown("""
<style>
    .main { background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); font-family: 'Arial', sans-serif; }
    .sidebar .sidebar-content { background: linear-gradient(180deg, #2C3E50, #3498DB); color: white; }
    .stButton > button { background: linear-gradient(45deg, #FF6B6B, #4ECDC4); color: white; border: none; border-radius: 25px; padding: 0.6rem 1.5rem; font-weight: bold; transition: all 0.3s ease; box-shadow: 0 4px 15px 0 rgba(31, 38, 135, 0.37); }
    .stButton > button:hover { transform: translateY(-2px); box-shadow: 0 8px 25px 0 rgba(31, 38, 135, 0.37); }
    .metric-container { background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 15px; padding: 1rem; margin: 0.5rem 0; border: 1px solid rgba(255, 255, 255, 0.2); }
    .main-header { text-align: center; padding: 2rem 0; background: rgba(255, 255, 255, 0.1); backdrop-filter: blur(10px); border-radius: 20px; margin-bottom: 2rem; border: 1px solid rgba(255, 255, 255, 0.2); }
    .stSuccess, .stError, .stWarning { border-radius: 10px; border: none; }
</style>
""", unsafe_allow_html=True)

# --- Main Header Content ---
st.markdown("""
<div class="main-header">
    <h1 style="color: white; font-size: 3rem; margin-bottom: 0;">Life Expectancy Prediction 🫀</h1>
    <p style="color: rgba(255,255,255,0.8); font-size: 1.2rem;">
        Machine Learning Pipeline for WHO Life Expectancy Dataset
    </p>
</div>
""", unsafe_allow_html=True)

# --- User Authentication Function ---
def check_authentication():
    if 'authenticated' not in st.session_state:
        st.session_state.authenticated = False
    
    if not st.session_state.authenticated:
        with st.sidebar:
            st.header("🔒 Authentication")
            password = st.text_input("Enter Password", type="password", key="auth_password")
            if st.button("🔑 Login", key="login_btn"):
                if password == "ds4everyone":
                    st.session_state.authenticated = True
                    st.success("✅ Access Granted!")
                    st.rerun()
                else:
                    st.error("❌ Incorrect Password")
        st.info("🔐 Please authenticate to access the application")
        st.stop() # Stops rendering further until authentication is successful

# Call the authentication function to secure the app
check_authentication()

# --- DagsHub and MLflow Initialization ---
if MLFLOW_AVAILABLE:
    try:
        # Set up MLflow tracking (local file-based storage first, then DagsHub config)
        mlflow.set_tracking_uri("file:///workspaces/pycaret-life-expectancy/mlruns")
        mlflow.set_experiment("life_expectancy_regression")
        
        # Ensure dagshub is imported locally within this block too if needed
        import dagshub 
        # dagshub.init(repo_owner='lh3594', repo_name='DSfinal', mlflow=True)
        dagshub.init(repo_owner='Tianjun-li-123', repo_name='DS4E-LIFE-EXP', mlflow=True)
        st.sidebar.success("✅ DagsHub and MLflow initialized.")
    except Exception as e:
        st.sidebar.error(f"❌ MLflow/DagsHub initialization failed: {str(e)}")
        # Provide guidance if token might be missing or permission issue
        if "AuthenticationError" in str(e) or "403" in str(e) or "401" in str(e):
             st.sidebar.warning(
                 "💡 DagsHub authentication failed. Please ensure your `DAGSHUB_USER_TOKEN` "
                 "environment variable is set with a valid DagsHub token that has write access to your repository. "
                 "You can generate one at `dagshub.com/user/settings/tokens`."
             )
        elif "unsupported endpoint" in str(e):
            st.sidebar.warning(
                "💡 DagsHub MLflow endpoint might not support all direct model logging features. "
                "The application attempts to use a workaround by logging models as `.joblib` artifacts. "
                "If issues persist, please refer to DagsHub documentation or contact their support."
            )
        else:
            st.sidebar.warning(f"💡 MLflow/DagsHub initialization encountered an unexpected error. Details: {e}")
else:
    st.sidebar.warning("MLflow not available. Experiment tracking disabled.")

# --- Data Loading and Initial Preprocessing ---
@st.cache_data(show_spinner="Loading and preparing data...")
def load_and_preprocess_data_cached():
    df = None
    st.info("Attempting to load data...")
    
    # Try DagsHub URL
    dagshub_url = "https://dagshub.com/lh3594/DSfinal/raw/main/Life%20Expectancy%20Data.csv"
    try:
        response = requests.get(dagshub_url, timeout=10)
        response.raise_for_status()
        df = pd.read_csv(io.StringIO(response.text))
        st.success("✅ Dataset loaded from DagsHub.")
    except Exception as e:
        st.warning(f"⚠️ Failed to download from DagsHub: {str(e)}. Trying local file.")
        
        # Try local file
        local_path = "Life Expectancy Data.csv"
        try:
            df = pd.read_csv(local_path)
            st.success("✅ Dataset loaded from local file.")
        except FileNotFoundError:
            st.warning(f"⚠️ Local file '{local_path}' not found. Trying public fallback URL.")
            
            # Try public fallback URL (WHO dataset from Kaggle)
            fallback_url = "https://raw.githubusercontent.com/dbouquin/IS_608/master/NHANES/who_life_exp.csv"
            try:
                response = requests.get(fallback_url, timeout=10)
                response.raise_for_status()
                df = pd.read_csv(io.StringIO(response.text))
                st.success("✅ Dataset loaded from public fallback URL.")
            except Exception as e:
                st.error(f"❌ Failed to load dataset from all sources: {str(e)}. "
                         "Please ensure 'Life Expectancy Data.csv' is available on DagsHub, "
                         "locally in the project root, or check internet connection.")
                return None, None, None, None, None, None, None, None # Return Nones if data not loaded

    if df is not None:
        # Limit for performance and sample consistently
        initial_df_length = len(df)
        if initial_df_length > 2000:
            df_sampled = df.sample(n=2000, random_state=42).reset_index(drop=True)
            st.info(f"📊 Dataset reduced from {initial_df_length} to 2000 samples for performance.")
        else:
            df_sampled = df.copy() # Use a copy to avoid modifying cache
        
        # Apply initial cleaning using the clean_missing function
        df_cleaned = clean_missing(df_sampled.copy(), numeric_strategy="median")
        st.info(f"🧹 Data cleaned: {initial_df_length - len(df_cleaned)} rows dropped due to NaNs after cleaning.")


        # Define features and target *after* cleaning and dropping 'Country'
        if "Life expectancy " not in df_cleaned.columns:
            st.error("Target column 'Life expectancy ' not found after preprocessing. Please check data file.")
            return None, None, None, None, None, None, None, None

        X_cols = [col for col in df_cleaned.columns if col not in ["Life expectancy ", "Country"]]
        X = df_cleaned[X_cols]
        y = df_cleaned["Life expectancy "]

        if not X.empty and not y.empty:
            scaler = StandardScaler()
            X_scaled_all = scaler.fit_transform(X) 
            X_train, X_test, y_train, y_test = train_test_split(X_scaled_all, y, test_size=0.2, random_state=42)
            
            return df_sampled, df_cleaned, X_train, X_test, y_train, y_test, X_cols, scaler
        else:
            st.error("Processed data (X or y) is empty after cleaning. Cannot proceed.")
            return None, None, None, None, None, None, None, None

    return None, None, None, None, None, None, None, None # Fallback if initial df is None


# --- Session State Initialization (All variables centralized here) ---
# This ensures all necessary state variables exist before pages try to access them.
if 'df_raw' not in st.session_state:
    st.session_state.df_raw = None # Stores the raw loaded DataFrame (sampled)
if 'df_cleaned' not in st.session_state:
    st.session_state.df_cleaned = None # Stores the main preprocessed DataFrame
if 'X_train' not in st.session_state:
    st.session_state.X_train = None
if 'X_test' not in st.session_state:
    st.session_state.X_test = None
if 'y_train' not in st.session_state:
    st.session_state.y_train = None
if 'y_test' not in st.session_state:
    st.session_state.y_test = None
if 'features' not in st.session_state:
    st.session_state.features = None # Stores list of feature names used for training
if 'scaler' not in st.session_state:
    st.session_state.scaler = None # Stores the fitted StandardScaler

if 'trained_models' not in st.session_state:
    st.session_state.trained_models = {} # Stores trained classical ML models
if 'classical_ml_metrics' not in st.session_state:
    st.session_state.classical_ml_metrics = {} # Stores metrics for classical ML models

if 'pycaret_final_model' not in st.session_state:
    st.session_state.pycaret_final_model = None # Stores the best model from PyCaret AutoML
if 'pycaret_comparison_df' not in st.session_state:
    st.session_state.pycaret_comparison_df = None # Stores PyCaret compare_models results
if 'pycaret_selected_features' not in st.session_state: # For PyCaret Explainability
    st.session_state.pycaret_selected_features = []
if 'automl_target_select' not in st.session_state: # For PyCaret Explainability
    st.session_state.automl_target_select = None


# Load and preprocess data only once and store in session state
if st.session_state.df_cleaned is None: # Only run if data isn't already loaded
    (st.session_state.df_raw, 
     st.session_state.df_cleaned, 
     st.session_state.X_train, 
     st.session_state.X_test, 
     st.session_state.y_train, 
     st.session_state.y_test, 
     st.session_state.features,
     st.session_state.scaler) = load_and_preprocess_data_cached()

# --- Utility Function: Get Dataset Information ---
def get_dataset_info(df_input):
    if df_input is None:
        return {'shape': (0,0), 'columns': [], 'dtypes': {}, 'missing_values': {}, 'memory_usage': "0.00 MB"}
    info = {
        'shape': df_input.shape,
        'columns': df_input.columns.tolist(),
        'dtypes': {str(k): str(v) for k,v in df_input.dtypes.to_dict().items()},
        'missing_values': df_input.isnull().sum().to_dict(),
        'memory_usage': f"{df_input.memory_usage(deep=True).sum() / 1024**2:.2f} MB"
    }
    return info


# --- Page Functions (Each function represents a "page" in the app) ---

def show_introduction_page():
    st.header("✨ Introduction")
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea, #764ba2); color: white; border-radius: 10px;">
        <h1>🫀 Life Expectancy Prediction</h1>
        <p style="font-size: 1.2rem;">
            This application predicts life expectancy using health and socioeconomic factors from the WHO Life Expectancy dataset. 
            Explore trends, compare predictive models, and understand key drivers of life expectancy through interactive visualizations and explainability tools.
        </p>
    </div>
    """, unsafe_allow_html=True)

    # Try to display an image (e.g., life.jpg in your project root)
    try:
        if os.path.exists("life.jpg"): # Assuming image is in the root or a known path
            st.image("life.jpg", use_column_width=True, caption="Factors influencing Life Expectancy")
        else:
            st.warning("Image 'life.jpg' not found in the project root. Please ensure it's there for display.")
    except Exception as e:
        st.warning(f"Could not load image: {e}")

    st.subheader("Dataset Overview")

    if st.session_state.df_raw is not None:
        st.write("The raw dataset includes features like immunization rates, income, education, and health metrics across countries.")
        
        st.markdown("---")
        st.subheader("Sample of Raw Data (initial load)")
        num_rows_raw = st.slider("Rows to display (Raw Data)", 1, min(100, len(st.session_state.df_raw)), 10, key="raw_rows_slider")
        st.dataframe(st.session_state.df_raw.head(num_rows_raw), use_container_width=True)
        
        st.markdown("---")
        st.subheader("Statistical Summary (Raw Data)")
        st.dataframe(st.session_state.df_raw.describe(), use_container_width=True)

        st.markdown("---")
        st.subheader("Missing Values (Raw Data)")
        missing_raw = st.session_state.df_raw.isnull().sum()
        st.dataframe(missing_raw[missing_raw > 0].to_frame("Missing Values (Raw)"), use_container_width=True)
        if missing_raw.sum() == 0:
            st.info("No missing values in raw data.")

        st.markdown("---")
        st.subheader("Sample of Cleaned Data (after preprocessing)")
        if st.session_state.df_cleaned is not None:
            num_rows_clean = st.slider("Rows to display (Cleaned Data)", 1, min(100, len(st.session_state.df_cleaned)), 10, key="clean_rows_slider")
            st.dataframe(st.session_state.df_cleaned.head(num_rows_clean), use_container_width=True)

            st.download_button(
                label="⬇️ Download Cleaned Data",
                data=st.session_state.df_cleaned.to_csv(index=False).encode('utf-8'),
                file_name="life_expectancy_cleaned.csv",
                mime="text/csv"
            )
        else:
            st.warning("Cleaned data is not available. Please check the data loading process.")

    else:
        st.warning("Data not loaded. Please ensure `Life Expectancy Data.csv` is accessible as configured.")


def show_data_exploration_page():
    st.header("🔍 Data Exploration")

    if st.session_state.df_cleaned is None:
        st.warning("⚠️ Cleaned data is not available. Please ensure data is loaded and processed on the Home page.")
        st.stop()

    df_explore = st.session_state.df_cleaned.copy()

    st.subheader("Statistical Summary")
    st.dataframe(df_explore.describe(), use_container_width=True)

    st.subheader("Missing Values (After Cleaning)")
    missing_cleaned = df_explore.isnull().sum()
    if missing_cleaned.sum() > 0:
        st.dataframe(missing_cleaned[missing_cleaned > 0].to_frame("Missing Values After Cleaning"), use_container_width=True)
        st.warning("Some missing values might still be present after initial cleaning. This might require more advanced imputation.")
        
        # Heatmap of missing values
        fig, ax = plt.subplots(figsize=(10, 6))
        sns.heatmap(df_explore.isnull(), cbar=False, cmap="viridis", ax=ax)
        ax.set_title("Heatmap of Missing Values (After Cleaning)")
        st.pyplot(fig)
        plt.close(fig) # Close figure to prevent memory issues
    else:
        st.success("✅ No missing values found in the cleaned dataset!")

    st.subheader("Data Types")
    st.dataframe(df_explore.dtypes.to_frame("Data Type"), use_container_width=True)

    st.subheader("Unique Values per Column")
    unique_counts = df_explore.nunique().to_frame("Unique Values")
    st.dataframe(unique_counts, use_container_width=True)


def show_visualization_page():
    st.header("📈 Data Visualization")

    if st.session_state.df_cleaned is None:
        st.warning("⚠️ Cleaned data is not available. Please ensure data is loaded and processed on the Home page.")
        st.stop()

    df_viz = st.session_state.df_cleaned.copy()
    target_column = "Life expectancy " # Assuming this is always the target

    st.subheader("Correlation Matrix")
    corr = df_viz.corr(numeric_only=True)
    fig_corr = px.imshow(corr, text_auto=True, color_continuous_scale="RdBu_r", title="Correlation Heatmap of Features",
                          width=800, height=700)
    st.plotly_chart(fig_corr, use_container_width=True)

    st.subheader("Distributions of Key Variables")
    selected_dist_col = st.selectbox("Select a column to view its distribution:", df_viz.columns, key="dist_col")
    if pd.api.types.is_numeric_dtype(df_viz[selected_dist_col]):
        fig_hist = px.histogram(df_viz, x=selected_dist_col, marginal="box", 
                                title=f"Distribution of {selected_dist_col}",
                                template="plotly_white")
        st.plotly_chart(fig_hist, use_container_width=True)
    else:
        counts = df_viz[selected_dist_col].value_counts().reset_index()
        counts.columns = [selected_dist_col, 'Count']
        fig_bar = px.bar(counts, x=selected_dist_col, y='Count', 
                         title=f"Distribution of {selected_dist_col}",
                         template="plotly_white")
        st.plotly_chart(fig_bar, use_container_width=True)


    st.subheader(f"Relationship with {target_column}")
    col_options = [col for col in df_viz.columns if col != target_column]
    if not col_options:
        st.warning("No other columns available for scatter plot after excluding target.")
    else:
        x_var = st.selectbox("Select X-axis variable:", col_options, key="scatter_x")
        fig_scatter = px.scatter(df_viz, x=x_var, y=target_column, 
                                 title=f"{target_column} vs {x_var}", 
                                 trendline="ols",
                                 color_discrete_sequence=px.colors.qualitative.Plotly,
                                 template="plotly_white")
        st.plotly_chart(fig_scatter, use_container_width=True)

    st.subheader("Pair Plot (Sampled)")
    st.write("Displays relationships between a few selected numerical features (sampling for performance).")
    num_cols = df_viz.select_dtypes(include=np.number).columns.tolist()
    if len(num_cols) > 1:
        pair_cols = st.multiselect("Select features for pair plot:", 
                                   num_cols, 
                                   default=num_cols[:min(5, len(num_cols))],
                                   key="pair_plot_cols")
        if pair_cols and len(pair_cols) > 1: # Ensure at least two columns selected for pairplot
            # Sample data for pair plot if dataset is large
            sample_size_pair = min(1000, len(df_viz))
            df_pair_sample = df_viz[pair_cols].sample(n=sample_size_pair, random_state=42)
            fig_pair = sns.pairplot(df_pair_sample)
            st.pyplot(fig_pair)
            plt.close(fig_pair.fig) # Corrected line: Access the underlying Figure object
        else:
            st.info("Select at least two numerical features for a pair plot.")
    else:
        st.info("Not enough numerical columns to generate a pair plot.")


def show_classical_ml_page():
    st.header("🤖 Classical Machine Learning")

    # --- Data Availability Check ---
    if st.session_state.df_cleaned is None or \
       st.session_state.X_train is None or \
       st.session_state.X_test is None or \
       st.session_state.y_train is None or \
       st.session_state.y_test is None or \
       st.session_state.features is None or \
       st.session_state.scaler is None:
        st.warning("⚠️ Data is not ready. Please ensure data is loaded and preprocessed on the Home page.")
        return # Use return instead of st.stop() in page functions

    # Retrieve preprocessed data from session state
    df_cleaned = st.session_state.df_cleaned
    X_train = st.session_state.X_train
    X_test = st.session_state.X_test
    y_train = st.session_state.y_train
    y_test = st.session_state.y_test
    features = st.session_state.features
    scaler = st.session_state.scaler

    st.subheader("Model Training")
    st.write("Train traditional machine learning models (Linear Regression, Random Forest) for life expectancy prediction. Metrics will be logged to MLflow.")

    if st.button("🚀 Train Classical Models", key="train_classical_models_btn"):
        with st.spinner("Training models... This might take a few moments for Random Forest with GridSearchCV."):
            st.session_state.trained_models, st.session_state.classical_ml_metrics = \
                train_classical_models(X_train, X_test, y_train, y_test, features)
        st.success("✅ Classical models training complete!")
    else:
        if not st.session_state.trained_models:
            st.info("Click 'Train Classical Models' to start training and view results.")


    # --- Model Evaluation ---
    if st.session_state.trained_models:
        st.subheader("Model Evaluation")
        
        classical_metrics_df = pd.DataFrame(st.session_state.classical_ml_metrics).T
        st.dataframe(classical_metrics_df, use_container_width=True)

        model_choice = st.selectbox("Select a trained model for detailed view:", 
                                    list(st.session_state.trained_models.keys()), 
                                    key="classical_model_select")
        
        if model_choice:
            selected_model = st.session_state.trained_models[model_choice]
            y_pred = selected_model.predict(X_test)
            st.write(f"**Selected Model: {model_choice}**")
            st.write(f"Metrics for {model_choice}:")
            st.json(st.session_state.classical_ml_metrics[model_choice])

            # Plot Actual vs Predicted
            fig_pred = px.scatter(x=y_test, y=y_pred, 
                                  labels={"x": "Actual Life Expectancy", "y": "Predicted Life Expectancy"}, 
                                  title=f"Actual vs Predicted Values ({model_choice})",
                                  template="plotly_white")
            fig_pred.add_shape(type="line", x0=y_test.min(), y0=y_test.min(), 
                              x1=y_test.max(), y1=y_test.max(), line=dict(color="red", dash="dash"))
            st.plotly_chart(fig_pred, use_container_width=True)

            st.subheader("Make a Prediction (Classical Models)")
            st.write("Enter values for features to get a prediction from the selected model.")
            
            input_data_dict = {}
            # Get mean values for pre-filling, use original (unscaled) df_cleaned for this
            for feature in features:
                # Ensure feature exists in df_cleaned before trying to get its mean
                if feature in df_cleaned.columns: 
                    default_value = float(df_cleaned[feature].mean())
                else:
                    default_value = 0.0 # Fallback for features not found (shouldn't happen if `features` is correct)
                input_data_dict[feature] = st.number_input(f"Enter {feature}", value=default_value, key=f"input_{feature}")
            
            user_input_df = pd.DataFrame([input_data_dict])
            
            try:
                # Scale user input using the same scaler fitted on training data
                input_scaled = scaler.transform(user_input_df)
                user_prediction = selected_model.predict(input_scaled)
                st.success(f"Predicted Life Expectancy ({model_choice}): **{user_prediction[0]:.2f} years**")
            except Exception as e:
                st.error(f"Error making prediction: {e}. Please ensure all input fields are valid.")
    else:
        st.info("No classical models have been trained yet.")


def show_pycaret_automl_page():
    st.header("⚡ PyCaret AutoML")

    # --- Data Availability Check ---
    if st.session_state.df_cleaned is None:
        st.warning("⚠️ Cleaned data is not available. Please ensure data is loaded and processed on the Home page.")
        return

    if not PYCARET_AVAILABLE:
        st.error("PyCaret is not installed. This page will not function.")
        return

    df_pycaret = st.session_state.df_cleaned.copy()
    target_column = "Life expectancy " # Target for PyCaret

    st.subheader("PyCaret Setup & Model Comparison")
    st.write("Run PyCaret's AutoML to automatically train and compare various regression models.")

    # Target and Feature Selection
    col1, col2 = st.columns(2)
    with col1:
        pycaret_target = st.selectbox(
            "🎯 Select target variable:",
            df_pycaret.columns,
            index=df_pycaret.columns.get_loc(target_column) if target_column in df_pycaret.columns else 0,
            key="pycaret_target_select_page"
        )
        # Filter out the target column from available features.
        available_features = [col for col in df_pycaret.columns if col != pycaret_target]
        
        # Allow multi-selection of features. Default to all available features except 'Country' (which should be dropped by cleaning)
        default_features = [c for c in available_features if c not in ["Country"]] 
        pycaret_features = st.multiselect(
            "📊 Select features:",
            available_features,
            default=default_features,
            key="pycaret_features_select_page"
        )

    with col2:
        # Option to use a sample of the data for faster processing.
        sample_size = st.slider("📊 Sample size for PyCaret:", 
                                500, min(5000, len(df_pycaret)), 1000, 
                                key="pycaret_sample_size_slider_page")
        if len(df_pycaret) > sample_size:
            model_df_pycaret = df_pycaret[pycaret_features + [pycaret_target]].sample(n=sample_size, random_state=42)
            st.info(f"📊 Using {len(model_df_pycaret)} samples for PyCaret setup.")
        else:
            model_df_pycaret = df_pycaret[pycaret_features + [pycaret_target]].copy()
            st.info("📊 Using full dataset for PyCaret setup.")

    # Drop any rows with NaN in the target column from the *sampled* DataFrame
    initial_rows_sampled = len(model_df_pycaret)
    model_df_pycaret.dropna(subset=[pycaret_target], inplace=True)
    if len(model_df_pycaret) < initial_rows_sampled:
        st.warning(f"Dropped {initial_rows_sampled - len(model_df_pycaret)} rows with missing target values in sampled data.")


    if st.button("🚀 Run PyCaret AutoML", key="run_pycaret_automl_btn"):
        if model_df_pycaret.empty:
            st.error("No data available for PyCaret setup after sampling/filtering. Please adjust selections.")
            return
        
        if not pycaret_features:
            st.warning("Please select at least one feature for PyCaret.")
            return

        with st.spinner("Setting up PyCaret environment and comparing models... This may take a while."):
            try:
                # PyCaret Setup: prepares the environment for modeling
                pycaret_env = reg_setup(
                    data=model_df_pycaret,
                    target=pycaret_target,
                    session_id=42, # For reproducibility
                    html=False, # Prevents PyCaret from trying to render interactive plots directly
                    # REMOVED: silent=True (deprecated/removed in newer PyCaret)
                    # REMOVED: verbose=False (deprecated/removed in newer PyCaret)
                    log_experiment=MLFLOW_AVAILABLE, # Log to MLflow if available
                    experiment_name="pycaret_life_expectancy" # Name for MLflow experiment
                )

                # Compare Models: Trains and evaluates multiple models, selecting the best one based on R2.
                best_model_pycaret = reg_compare(sort="R2", n_select=1)

                if best_model_pycaret is None or (isinstance(best_model_pycaret, list) and not best_model_pycaret):
                    st.error("❌ PyCaret could not train or identify a best model. Please check your data, features, and target.")
                    return
                
                best_pycaret_model_obj = best_model_pycaret[0] if isinstance(best_model_pycaret, list) else best_model_pycaret

                # Finalize Model: Trains the best model on the entire dataset
                finalized_pycaret_model = reg_finalize(best_pycaret_model_obj)
                st.session_state.pycaret_final_model = finalized_pycaret_model # Store finalized model in session state

                # Store features and target used by PyCaret for Explainability page
                st.session_state.pycaret_selected_features = pycaret_features
                st.session_state.automl_target_select = pycaret_target


                # Pull comparison results for display
                comparison_df_pycaret = reg_pull()
                st.session_state.pycaret_comparison_df = comparison_df_pycaret # Store for later use

                # Save the best model locally
                model_save_path = "best_pycaret_model.pkl"
                reg_save_model(finalized_pycaret_model, model_save_path)
                st.info(f"Model saved locally to {model_save_path}")

                if MLFLOW_AVAILABLE:
                    # Log the saved model as an MLflow artifact outside of the PyCaret's run (if PyCaret's log_experiment is not enough)
                    # Note: PyCaret's setup will create an MLflow run. This is for explicit logging outside of that if needed.
                    with mlflow.start_run(run_name="PyCaret_Final_Model_Log", nested=True) as run:
                        mlflow.log_artifact(model_save_path)
                        st.success(f"PyCaret best model logged as artifact to MLflow Run ID: {run.info.run_id}")


                st.success("✅ PyCaret AutoML training complete!")

                st.subheader("📈 PyCaret Model Comparison Results")
                st.dataframe(comparison_df_pycaret[["Model", "MAE", "RMSE", "R2", "TT (Sec)"]], use_container_width=True)

                st.subheader("🏆 Best PyCaret Model Summary")
                st.write(f"**Best Model:** {best_pycaret_model_obj.name if hasattr(best_pycaret_model_obj, 'name') else type(best_pycaret_model_obj).__name__}")
                st.write(finalized_pycaret_model) # Display the finalized model object

                # --- Evaluate on Sampled Data ---
                st.subheader("📊 Performance of Finalized PyCaret Model")
                predictions_pycaret = reg_predict(finalized_pycaret_model, data=model_df_pycaret)
                
                actual_pycaret = predictions_pycaret[pycaret_target]
                predicted_pycaret = predictions_pycaret["prediction_label"]

                metrics_pycaret = {}
                metrics_pycaret["R2"] = r2_score(actual_pycaret, predicted_pycaret)
                metrics_pycaret["MAE"] = mean_absolute_error(actual_pycaret, predicted_pycaret)
                metrics_pycaret["RMSE"] = np.sqrt(mean_squared_error(actual_pycaret, predicted_pycaret))

                cols_metrics_pycaret = st.columns(3)
                for i, (name, val) in enumerate(metrics_pycaret.items()):
                    cols_metrics_pycaret[i].metric(name, f"{val:.4f}")

                # Plot Actual vs. Predicted values
                fig_pred_pycaret = px.scatter(
                    predictions_pycaret, 
                    x=actual_pycaret, 
                    y=predicted_pycaret, 
                    labels={'x': f'Actual {pycaret_target}', 'y': f'Predicted {pycaret_target}'},
                    title=f'Actual vs Predicted Values (PyCaret Best Model)',
                    template="plotly_white"
                )
                fig_pred_pycaret.add_shape(type="line", x0=min(actual_pycaret), y0=min(actual_pycaret), 
                                          x1=max(actual_pycaret), y1=max(actual_pycaret), line=dict(color="red", dash="dash"))
                st.plotly_chart(fig_pred_pycaret, use_container_width=True)

                st.subheader("🔮 Sample PyCaret Predictions")
                st.dataframe(predictions_pycaret[[pycaret_target, "prediction_label"]].head(10))
                
            except Exception as e:
                st.error(f"❌ An error occurred during PyCaret operations: {str(e)}. "
                         "Please check your data, features, target, and PyCaret installation.")
                st.exception(e)
    else:
        st.info("Click 'Run PyCaret AutoML' to begin the automated machine learning process.")

    st.subheader("Existing PyCaret Model & Results")
    if st.session_state.pycaret_final_model is not None:
        st.write("A PyCaret model has been previously trained and finalized.")
        st.write(st.session_state.pycaret_final_model)
        
        if st.session_state.pycaret_comparison_df is not None:
            st.write("Previous Model Comparison Results:")
            st.dataframe(st.session_state.pycaret_comparison_df[["Model", "MAE", "RMSE", "R2", "TT (Sec)"]], use_container_width=True)
    else:
        st.info("No PyCaret model has been finalized in this session yet.")


def show_explainability_page():
    st.header("🔬 Model Explainability (SHAP)")

    if not SHAP_AVAILABLE:
        st.error("SHAP library is not installed. Please install it (`pip install shap`) to use this page.")
        return

    if st.session_state.df_cleaned is None or \
       st.session_state.X_test is None or \
       st.session_state.features is None:
        st.warning("⚠️ Data is not ready. Please ensure data is loaded and preprocessed on the Home page.")
        return

    # Retrieve data from session state
    X_test = st.session_state.X_test
    features = st.session_state.features

    # Combine classical and PyCaret models for selection
    all_models = {}
    if st.session_state.trained_models:
        all_models.update(st.session_state.trained_models)
    if st.session_state.pycaret_final_model:
        all_models["PyCaret Best Model"] = st.session_state.pycaret_final_model

    if not all_models:
        st.warning("No models available for SHAP analysis. Train models on 'Classical ML' or 'PyCaret AutoML' page first.")
        return

    model_choice = st.selectbox("Select Model for SHAP Analysis", list(all_models.keys()), key="shap_model_select")

    if model_choice:
        selected_model = all_models[model_choice]

        st.subheader(f"SHAP Analysis for: {model_choice}")
        
        # Generate SHAP values (use a subset of X_test for performance)
        shap_values, feature_names_for_shap = shap_explain(selected_model, X_test, features)

        if shap_values is None:
            st.error("Failed to generate SHAP values. Check console for errors from `shap_explain` function.")
            return

        # Ensure feature_names is correctly set for plotting
        if isinstance(shap_values, shap.Explanation):
            if shap_values.feature_names is None and feature_names_for_shap is not None:
                shap_values.feature_names = list(feature_names_for_shap) # Ensure it's a list for shap

        if shap_values is not None:
            st.subheader("Global Feature Importance (Mean Absolute SHAP Value)")
            try:
                fig_bar, ax_bar = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, plot_type="bar", show=False)
                ax_bar.set_title("Global Feature Importance")
                st.pyplot(fig_bar, use_container_width=True)
                plt.close(fig_bar) # Close figure
            except Exception as e:
                st.error(f"Error plotting SHAP summary (bar): {e}")
                st.exception(e)

            st.subheader("Feature Impact Summary (SHAP Beeswarm Plot)")
            st.write("Each point is an observation. Its horizontal position shows the impact on the model's output. Color shows feature value.")
            try:
                fig_beeswarm, ax_beeswarm = plt.subplots(figsize=(10, 6))
                shap.summary_plot(shap_values, show=False)
                ax_beeswarm.set_title("Feature Impact Summary (Beeswarm Plot)")
                st.pyplot(fig_beeswarm, use_container_width=True)
                plt.close(fig_beeswarm) # Close figure
            except Exception as e:
                st.error(f"Error plotting SHAP summary (beeswarm): {e}")
                st.exception(e)

            st.subheader("Individual Prediction Explanation (Waterfall Plot)")
            st.write("Understand how individual features contribute to a single prediction.")
            
            if len(X_test) > 0:
                instance_idx = st.slider("Select an instance to explain (0-indexed):", 0, len(X_test) - 1, 0, key="shap_instance_slider")
                try:
                    # Need to provide the specific SHAP values for that instance
                    single_shap_value = shap_values[instance_idx]
                    
                    fig_waterfall, ax_waterfall = plt.subplots(figsize=(10, 7))
                    shap.plots.waterfall(single_shap_value, show=False)
                    ax_waterfall.set_title(f"Explanation for Instance {instance_idx}")
                    st.pyplot(fig_waterfall, use_container_width=True)
                    plt.close(fig_waterfall) # Close figure
                except Exception as e:
                    st.error(f"Error plotting SHAP waterfall for instance {instance_idx}: {e}")
                    st.exception(e)
            else:
                st.info("No test instances available for individual prediction explanation.")
        else:
            st.warning("SHAP values could not be generated. Check previous error messages.")


def show_mlflow_tracking_page():
    st.header("📋 MLflow Experiment Tracking")
    st.write("This page shows the results of machine learning experiments tracked with MLflow.")

    if not MLFLOW_AVAILABLE:
        st.warning("MLflow is not available. Experiment tracking features are disabled.")
        return

    # Ensure the MLflow tracking URI is set
    try:
        current_tracking_uri = mlflow.get_tracking_uri()
        if not current_tracking_uri:
             # Fallback if somehow not set, though main app should set it
             mlflow.set_tracking_uri("file:///workspaces/pycaret-life-expectancy/mlruns")
        st.info(f"MLflow is tracking to: `{mlflow.get_tracking_uri()}`")
    except Exception as e:
        st.error(f"Could not set MLflow tracking URI: {e}. Ensure MLflow is installed and configured.")
        return

    st.subheader("View All Runs")
    st.write("You can also view these runs in the MLflow UI by running `mlflow ui` in your terminal inside your project directory.")

    try:
        # Fetch all runs for the experiment
        experiment_name = "life_expectancy_regression" # Matches the name in streamlit_app.py
        runs_df_original = mlflow.search_runs(experiment_names=[experiment_name], order_by=["metrics.R2 DESC"])

        if not runs_df_original.empty:
            # Rename columns and assign back to runs_df
            runs_df = runs_df_original.rename(columns={
                "tags.mlflow.runName": "Run Name",
                "metrics.R2": "R2",
                "metrics.MAE": "MAE",
                "metrics.MSE": "MSE",
                "params.model": "Model Type",
                "params.n_estimators": "RF N Estimators",
                "params.max_depth": "RF Max Depth"
            })

            # Display key run information
            st.dataframe(runs_df[[
                "start_time", 
                "Run Name",  # Use the renamed column
                "R2", 
                "MAE", 
                "MSE", 
                "Model Type",
                "RF N Estimators", 
                "RF Max Depth"     
            ]], use_container_width=True)

            st.subheader("R2 Score Comparison Across Runs")
            # Ensure 'Run Name' column exists for plotting
            if 'Run Name' in runs_df.columns: # This check is now robust as runs_df is updated
                fig_r2 = px.bar(runs_df, x='Run Name', y='R2', 
                                title='R2 Score Comparison',
                                color='R2', color_continuous_scale=px.colors.sequential.Viridis)
                st.plotly_chart(fig_r2, use_container_width=True)
            else:
                st.warning("Could not find 'Run Name' column for R2 comparison plot.")


            st.subheader("Detailed Run Information")
            # This line will now correctly find 'Run Name'
            selected_run_name = st.selectbox("Select a specific run to view details:", runs_df["Run Name"].tolist())
            if selected_run_name:
                selected_run = runs_df_original[runs_df_original["tags.mlflow.runName"] == selected_run_name].iloc[0] # Use original df for full data
                st.json(selected_run.to_dict())

                # Optionally, retrieve and display artifacts for the selected run
                st.subheader("Run Artifacts")
                try:
                    # Assuming the MLflow run_id is available in the DataFrame
                    run_id_for_artifacts = selected_run["run_id"] # Use run_id from the original selected run
                    # Create an MLflowClient instance for artifact listing
                    client = mlflow.tracking.MlflowClient() 
                    artifacts = client.list_artifacts(run_id=run_id_for_artifacts)
                    artifact_names = [a.path for a in artifacts]
                    if artifact_names:
                        st.write("Available artifacts:")
                        for artifact_name in artifact_names:
                            st.write(f"- {artifact_name}")
                            # You can add logic here to download/display specific artifacts
                    else:
                        st.info("No artifacts logged for this run.")
                except Exception as art_e:
                    st.error(f"Error listing artifacts for selected run: {art_e}")


        else:
            st.info("No MLflow runs found for the 'life_expectancy_regression' experiment. "
                    "Train models on the 'Classical ML' or 'PyCaret AutoML' pages to log runs.")

    except Exception as e:
        st.error(f"An error occurred while fetching MLflow runs: {str(e)}")
        st.write("Make sure your MLflow tracking server is accessible and experiments are being logged.")
        st.exception(e) # Display full traceback


# --- Sidebar Navigation ---
st.sidebar.title("🫀 Life Expectancy Dashboard")
page_selection = st.sidebar.selectbox("Select Page", 
                                     ["Home", "Introduction", "Data Exploration", "Visualization", 
                                      "Classical ML", "PyCaret AutoML", "Explainability", 
                                      "MLflow Tracking"])

# --- Call the selected page function ---
if page_selection == "Home":
    # The home page content is already displayed at the end of the main script
    # Display the "Home" page content here
    col1, col2, col3 = st.columns([1, 2, 1])
    with col2:
        st.markdown("""
        ## Welcome to the Life Expectancy Prediction App! 🎉
        This app provides a complete ML pipeline for predicting life expectancy using the WHO dataset:
        - Explore data insights on **📊 Data Exploration** and **📈 Visualization** pages.
        - Train classical ML models on **🤖 Classical ML** page.
        - Run AutoML experiments with PyCaret on **⚡ PyCaret AutoML** page.
        - Understand model decisions with **🔬 Explainability**.
        - Track all experiments via **📋 MLflow Tracking**.
        """)
        st.markdown("---")
        
        if st.session_state.df_cleaned is not None:
            info = get_dataset_info(st.session_state.df_cleaned)
            col_a, col_b, col_c = st.columns(3)
            with col_a:
                st.metric("📊 Cleaned Rows", f"{info['shape'][0]:,}")
            with col_b:
                st.metric("📋 Columns", f"{info['shape'][1]:,}")
            with col_c:
                st.metric("🤖 Models Trained", len(st.session_state.trained_models) + (1 if st.session_state.pycaret_final_model else 0))
            with st.expander("📋 Dataset Details (Cleaned Data)"):
                st.json(info)
            
            # Display sample of cleaned data on home page for quick glance
            st.subheader("Sample of Cleaned Data")
            st.dataframe(st.session_state.df_cleaned.head())

        else:
            st.warning("⚠️ Data is not loaded or could not be cleaned. Please check the console for errors.")

elif page_selection == "Introduction":
    show_introduction_page()
elif page_selection == "Data Exploration":
    show_data_exploration_page()
elif page_selection == "Visualization":
    show_visualization_page()
elif page_selection == "Classical ML":
    show_classical_ml_page()
elif page_selection == "PyCaret AutoML":
    show_pycaret_automl_page()
elif page_selection == "Explainability":
    show_explainability_page()
elif page_selection == "MLflow Tracking":
    show_mlflow_tracking_page()

# --- Footer ---
st.markdown("---")
st.write("**Life Expectancy Prediction App** | Built with Streamlit | © 2025")
