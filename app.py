import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from io import BytesIO
import pickle
pip install -r requirements.txt

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

st.set_page_config(page_title="Airbnb Price Predictor", layout="wide")

st.title("Airbnb Price Prediction — Full Pipeline (Train & Predict)")
st.write("Upload the Airbnb CSV, preprocess, train a Random Forest model, and predict prices interactively.")

# ------------------------------------------------------------------
# Helpers
# ------------------------------------------------------------------
@st.cache_data
def load_csv(uploaded_file):
    return pd.read_csv(uploaded_file)

@st.cache_data
def default_load(path='AB_NYC_2019.csv'):
    try:
        return pd.read_csv(path)
    except Exception:
        return None

def basic_preprocess(df):
    df = df.copy()

    # Standard quick checks
    if 'price' not in df.columns:
        raise ValueError("No 'price' column detected in the dataset.")

    # Remove obvious bad prices
    df = df[df['price'] >= 0]

    # Fill or create some commonly used features that exist in AB_NYC_2019
    # For robustness, only use columns if they exist
    # Convert last_review to datetime and extract year/month if exists
    if 'last_review' in df.columns:
        try:
            df['last_review'] = pd.to_datetime(df['last_review'], errors='coerce')
            df['last_review_year'] = df['last_review'].dt.year.fillna(0).astype(int)
            df['last_review_month'] = df['last_review'].dt.month.fillna(0).astype(int)
        except Exception:
            df['last_review_year'] = 0
            df['last_review_month'] = 0

    # Fill missing numeric features with 0 or median
    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    for col in ['reviews_per_month', 'number_of_reviews']:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    # Keep a sensible subset of features if present
    features = []
    candidates = [
        'neighbourhood_group', 'neighbourhood', 'room_type',
        'latitude', 'longitude', 'minimum_nights', 'number_of_reviews',
        'reviews_per_month', 'calculated_host_listings_count', 'availability_365',
        'last_review_year', 'last_review_month'
    ]
    for c in candidates:
        if c in df.columns:
            features.append(c)

    # Drop rows with missing price
    df = df.dropna(subset=['price'])

    return df, features

@st.cache_data
def build_pipeline(numeric_features, categorical_features, n_estimators, max_depth, random_state=42):
    numeric_transformer = Pipeline(steps=[('scaler', StandardScaler())])
    categorical_transformer = Pipeline(steps=[('ohe', OneHotEncoder(handle_unknown='ignore'))])

    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numeric_transformer, numeric_features),
            ('cat', categorical_transformer, categorical_features)
        ])

    model = RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth, random_state=random_state)

    pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('model', model)])
    return pipeline

# ------------------------------------------------------------------
# Sidebar: Upload / dataset options
# ------------------------------------------------------------------
st.sidebar.header("Dataset")
uploaded_file = st.sidebar.file_uploader("Upload CSV file (AB_NYC_2019 or similar)", type=['csv'])
use_default = False
if uploaded_file is None:
    # Try to load default CSV if it exists in working directory
    default_df = default_load()
    if default_df is not None:
        st.sidebar.success("Found default 'AB_NYC_2019.csv' in the working directory. Using that file.")
        use_default = True
    else:
        st.sidebar.info("No file uploaded. Upload a CSV or place 'AB_NYC_2019.csv' in the app directory.")

if uploaded_file is not None:
    try:
        df = load_csv(uploaded_file)
    except Exception as e:
        st.error(f"Failed to read uploaded file: {e}")
        st.stop()
elif use_default:
    df = default_df
else:
    st.stop()

st.sidebar.write(f"Rows: {df.shape[0]} — Columns: {df.shape[1]}")

# ------------------------------------------------------------------
# Data preview and preprocessing
# ------------------------------------------------------------------
st.header("1) Data preview & preprocessing")
with st.expander("Show raw data (first 100 rows)"):
    st.dataframe(df.head(100))

try:
    df_clean, features = basic_preprocess(df)
except Exception as e:
    st.error(f"Preprocessing failed: {e}")
    st.stop()

st.write(f"Detected features to use: **{features}**")

# Let user adjust target and features
target = st.selectbox("Target column (price)", options=['price'], index=0)
additional_features = st.multiselect("Add other columns to use as features (if present)", options=[c for c in df.columns if c not in features and c != 'price'])
for c in additional_features:
    if c not in features:
        features.append(c)

st.write("Final feature list:", features)

# ------------------------------------------------------------------
# Train-test split and model hyperparameters
# ------------------------------------------------------------------
st.header("2) Train model")
test_size = st.slider("Test set proportion", min_value=0.05, max_value=0.5, value=0.25, step=0.05)
random_state = st.number_input("Random state (seed)", min_value=0, max_value=9999, value=42, step=1)

n_estimators = st.slider("Random Forest: n_estimators", 50, 500, 100, step=50)
max_depth = st.slider("Random Forest: max_depth (0 = None)", 0, 50, 10, step=1)
max_depth_val = None if max_depth == 0 else max_depth

train_button = st.button("Train model")

if train_button:
    # Prepare X, y
    X = df_clean[features].copy()
    y = df_clean[target].astype(float)

    # Separate numeric and categorical features
    numeric_features = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_features = [c for c in X.columns if c not in numeric_features]

    st.write(f"Numeric features: {numeric_features}")
    st.write(f"Categorical features: {categorical_features}")

    pipeline = build_pipeline(numeric_features, categorical_features, n_estimators=n_estimators, max_depth=max_depth_val, random_state=random_state)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

    with st.spinner("Training model — this may take a while depending on n_estimators and data size..."):
        pipeline.fit(X_train, y_train)

    # Evaluate
    preds = pipeline.predict(X_test)
    mae = mean_absolute_error(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    r2 = r2_score(y_test, preds)

    st.success("Training completed")
    st.metric("MAE", f"{mae:,.2f}")
    st.metric("RMSE", f"{rmse:,.2f}")
    st.metric("R^2", f"{r2:.4f}")

    # Feature importances (approx)
    try:
        model = pipeline.named_steps['model']
        preprocessor = pipeline.named_steps['preprocessor']

        # Get feature names after preprocessing
        cat_cols = []
        if categorical_features:
            ohe = preprocessor.named_transformers_['cat'].named_steps['ohe']
            cat_cols = ohe.get_feature_names_out(categorical_features).tolist()
        feature_names = numeric_features + cat_cols

        importances = model.feature_importances_
        fi = pd.DataFrame({'feature': feature_names, 'importance': importances})
        fi = fi.sort_values('importance', ascending=False).head(30)

        st.subheader("Top feature importances")
        st.dataframe(fi)

        fig, ax = plt.subplots(figsize=(8, min(6, len(fi) * 0.3 + 1)))
        ax.barh(fi['feature'][::-1], fi['importance'][::-1])
        ax.set_xlabel('Importance')
        ax.set_ylabel('Feature')
        ax.set_title('Feature importances (top)')
        st.pyplot(fig)
    except Exception as e:
        st.info(f"Couldn't compute feature importances: {e}")

    # Save model to in-memory bytes for download
    model_bytes = BytesIO()
    pickle.dump(pipeline, model_bytes)
    model_bytes.seek(0)

    st.download_button(label="Download trained model (pickle)", data=model_bytes, file_name="airbnb_rf_pipeline.pkl", mime='application/octet-stream')

    # Persist pipeline in session state for prediction UI
    st.session_state['pipeline'] = pipeline
    st.session_state['features'] = features
    st.session_state['numeric_features'] = numeric_features
    st.session_state['categorical_features'] = categorical_features

# ------------------------------------------------------------------
# Prediction interface
# ------------------------------------------------------------------
st.header("3) Predict")
if 'pipeline' not in st.session_state:
    st.info("No trained model in this session. Train a model above or upload a pretrained pipeline pickle below.")

col1, col2 = st.columns([2, 1])
with col2:
    uploaded_model = st.file_uploader("Or upload a trained pipeline pickle (.pkl)", type=['pkl', 'pickle'])
    if uploaded_model is not None:
        try:
            loaded = pickle.load(uploaded_model)
            st.session_state['pipeline'] = loaded
            # Attempt to infer feature names from earlier saved session or from upload metadata
            st.success("Uploaded model loaded into session. Use the form below to predict.")
        except Exception as e:
            st.error(f"Failed to load pickle: {e}")

if 'pipeline' in st.session_state:
    pipeline = st.session_state['pipeline']
    features = st.session_state.get('features', None)
    numeric_features = st.session_state.get('numeric_features', None)
    categorical_features = st.session_state.get('categorical_features', None)

    st.subheader("Enter feature values to predict price")
    with st.form(key='predict_form'):
        input_data = {}
        # If we know features, create inputs for them
        if features is not None:
            for f in features:
                # numeric
                if f in numeric_features:
                    val = st.number_input(f, value=float(df_clean[f].median() if f in df_clean else 0.0))
                    input_data[f] = [val]
                else:
                    # for categorical, provide selectbox of top categories
                    if f in df_clean.columns:
                        top_vals = df_clean[f].dropna().astype(str).value_counts().nlargest(20).index.tolist()
                        if len(top_vals) == 0:
                            txt = st.text_input(f, value='')
                            input_data[f] = [txt]
                        else:
                            sel = st.selectbox(f, options=top_vals, index=0)
                            input_data[f] = [sel]
                    else:
                        txt = st.text_input(f, value='')
                        input_data[f] = [txt]
        else:
            st.info("Model does not expose feature list. Provide inputs matching the model's expected features by name.")
            st.text_area("JSON-like input (e.g. {\"latitude\": [40.7], \"longitude\": [-73.9], \"room_type\": [\"Entire home/apt\"]})")

        submit = st.form_submit_button("Predict")

    if submit:
        try:
            X_new = pd.DataFrame(input_data)
            pred = pipeline.predict(X_new)[0]
            st.success(f"Predicted price: {pred:,.2f}")
        except Exception as e:
            st.error(f"Prediction failed: {e}")

# ------------------------------------------------------------------
# Extras: simple EDA
# ------------------------------------------------------------------
st.header("4) Quick EDA")
if st.checkbox("Show price distribution"):
    fig, ax = plt.subplots()
    ax.hist(df_clean['price'].clip(0, df_clean['price'].quantile(0.99)), bins=50)
    ax.set_xlabel('Price (clipped at 99th percentile)')
    ax.set_ylabel('Count')
    st.pyplot(fig)

if st.checkbox("Show average price by neighbourhood_group (if present)"):
    if 'neighbourhood_group' in df_clean.columns:
        grp = df_clean.groupby('neighbourhood_group')['price'].median().sort_values()
        st.bar_chart(grp)
    else:
        st.info("Column 'neighbourhood_group' not found in dataset.")

st.write("---")
st.write("If you'd like, I can also generate a requirements.txt for deploying this app — tell me and I'll add it.")
