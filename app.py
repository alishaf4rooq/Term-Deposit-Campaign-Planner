
import streamlit as st
import pandas as pd
import numpy as np
import sqlite3
import joblib
from pathlib import Path

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report


# =========================================================
# PAGE CONFIGURATION
# =========================================================

st.set_page_config(
    page_title="Term Deposit Campaign Planner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# CUSTOM CSS - TEXT VISIBILITY FIX
# =========================================================

st.markdown("""
<style>

/* Main application */
.stApp {
    background-color: #f5f7fb;
}

/* Main headings */
h1, h2, h3, h4 {
    color: #172033 !important;
}

/* Normal text */
p, span, label, li {
    color: #263247;
}

/* Metric cards */
div[data-testid="stMetric"] {
    background-color: #ffffff !important;
    padding: 20px !important;
    border-radius: 12px !important;
    border: 1px solid #e2e8f0 !important;
    box-shadow: 0 2px 7px rgba(0,0,0,0.05);
}

/* Metric labels */
div[data-testid="stMetric"] [data-testid="stMetricLabel"],
div[data-testid="stMetric"] [data-testid="stMetricLabel"] p,
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] label p {
    color: #263247 !important;
    font-size: 15px !important;
    font-weight: 600 !important;
}

/* Metric numbers */
div[data-testid="stMetric"] [data-testid="stMetricValue"],
div[data-testid="stMetric"] [data-testid="stMetricValue"] div {
    color: #111827 !important;
    font-size: 30px !important;
    font-weight: 700 !important;
}

/* Metric delta */
div[data-testid="stMetric"] [data-testid="stMetricDelta"],
div[data-testid="stMetric"] [data-testid="stMetricDelta"] p {
    color: #334155 !important;
}

/* Cards and containers */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #ffffff;
    border-radius: 12px;
}

/* Dataframes */
div[data-testid="stDataFrame"] {
    background-color: #ffffff;
}

/* Sidebar */
section[data-testid="stSidebar"] {
    background-color: #172033;
}

section[data-testid="stSidebar"] * {
    color: #ffffff !important;
}

/* Buttons */
.stButton button {
    border-radius: 8px;
    font-weight: 600;
}

/* Inputs */
input, textarea {
    color: #172033 !important;
}

/* Success and information boxes */
div[data-testid="stAlert"] p {
    color: #172033 !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PATHS
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "bank-additional-full.csv"

# Fallback for a CSV temporarily uploaded in the root folder
ROOT_DATA_PATH = BASE_DIR / "bank-additional-full.csv"

MODEL_PATH = BASE_DIR / "model" / "bank_subscription_model.joblib"

DB_PATH = BASE_DIR / "campaigns.db"


# =========================================================
# DATABASE FUNCTIONS
# =========================================================

def initialize_database():

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS campaigns (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            campaign_name TEXT NOT NULL,
            target_segment TEXT,
            budget REAL,
            start_date TEXT,
            notes TEXT
        )
    """)

    connection.commit()
    connection.close()


def add_campaign(name, segment, budget, start_date, notes):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute("""
        INSERT INTO campaigns
        (campaign_name, target_segment, budget, start_date, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (name, segment, budget, start_date, notes))

    connection.commit()
    connection.close()


def get_campaigns():

    connection = sqlite3.connect(DB_PATH)

    df = pd.read_sql_query(
        "SELECT * FROM campaigns ORDER BY id DESC",
        connection
    )

    connection.close()

    return df


def delete_campaign(campaign_id):

    connection = sqlite3.connect(DB_PATH)

    cursor = connection.cursor()

    cursor.execute(
        "DELETE FROM campaigns WHERE id = ?",
        (campaign_id,)
    )

    connection.commit()
    connection.close()


initialize_database()


# =========================================================
# DATASET LOADING
# =========================================================

@st.cache_data
def load_dataset():

    if DATA_PATH.exists():
        path = DATA_PATH

    elif ROOT_DATA_PATH.exists():
        path = ROOT_DATA_PATH

    else:
        raise FileNotFoundError(
            "Dataset not found. Please upload "
            "bank-additional-full.csv inside the data folder."
        )

    df = pd.read_csv(path, sep=";")

    df.columns = df.columns.str.strip()

    return df


try:
    df = load_dataset()

except Exception as error:

    st.error(f"Dataset could not be loaded: {error}")

    st.info(
        "Check that data/bank-additional-full.csv "
        "exists and is the original semicolon-separated CSV."
    )

    st.stop()


# =========================================================
# GENERAL DATA PREPARATION
# =========================================================

df["y"] = df["y"].astype(str).str.lower().str.strip()

total_customers = len(df)

subscribed = (df["y"] == "yes").sum()

not_subscribed = (df["y"] == "no").sum()

subscription_rate = (
    subscribed / total_customers * 100
    if total_customers > 0 else 0
)


# =========================================================
# MACHINE LEARNING FUNCTIONS
# =========================================================

@st.cache_resource
def train_model():

    data = load_dataset().copy()

    X = data.drop(columns=["y", "duration"], errors="ignore")

    y = data["y"].astype(str).str.lower().str.strip()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="median"))
    ])

    categorical_pipeline = Pipeline([
        ("imputer", SimpleImputer(strategy="most_frequent")),
        ("encoder", OneHotEncoder(handle_unknown="ignore"))
    ])

    preprocessor = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    model = Pipeline([
        ("preprocessor", preprocessor),
        ("classifier", RandomForestClassifier(
            n_estimators=150,
            random_state=42,
            class_weight="balanced",
            n_jobs=-1
        ))
    ])

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y
    )

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    accuracy = accuracy_score(y_test, predictions)

    return model, accuracy, X.columns.tolist()


# =========================================================
# SIDEBAR NAVIGATION
# =========================================================

st.sidebar.title("📊 Term Deposit Planner")

st.sidebar.write("Campaign Management System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dashboard",
        "Campaign Planner",
        "Customer Insights",
        "Subscription Predictor",
        "Model & Dataset"
    ]
)

st.sidebar.divider()

st.sidebar.caption("Academic Machine Learning Project")

st.sidebar.caption("UCI Bank Marketing Dataset")


# =========================================================
# DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("📊 Dashboard")

    st.write(
        "Welcome to the Term-Deposit Campaign Planner. "
        "Analyze customers and plan effective marketing campaigns."
    )

    st.subheader("Campaign Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Subscribed Customers",
            f"{subscribed:,}"
        )

    with col3:
        st.metric(
            "Not Subscribed",
            f"{not_subscribed:,}"
        )

    with col4:
        st.metric(
            "Subscription Rate",
            f"{subscription_rate:.2f}%"
        )

    st.divider()

    st.subheader("Subscription Analysis")

    col1, col2 = st.columns(2)

    with col1:

        channel_data = (
            df.groupby("contact")["y"]
            .apply(lambda x: (x == "yes").mean() * 100)
            .reset_index(name="Subscription Rate")
        )

        fig = px.bar(
            channel_data,
            x="contact",
            y="Subscription Rate",
            title="Subscription Rate by Contact Channel",
            color="Subscription Rate",
            color_continuous_scale="Blues"
        )

        st.plotly_chart(fig, use_container_width=True)

    with col2:

        month_data = (
            df.groupby("month")["y"]
            .apply(lambda x: (x == "yes").mean() * 100)
            .reset_index(name="Subscription Rate")
        )

        fig = px.bar(
            month_data,
            x="month",
            y="Subscription Rate",
            title="Subscription Rate by Month",
            color="Subscription Rate",
            color_continuous_scale="Teal"
        )

        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Subscription Distribution")

    distribution = pd.DataFrame({
        "Subscription": ["Yes", "No"],
        "Customers": [subscribed, not_subscribed]
    })

    fig = px.pie(
        distribution,
        names="Subscription",
        values="Customers",
        hole=0.45,
        title="Customer Subscription Distribution"
    )

    st.plotly_chart(fig, use_container_width=True)


# =========================================================
# CAMPAIGN PLANNER
# =========================================================

elif page == "Campaign Planner":

    st.title("📋 Campaign Planner")

    st.write(
        "Create and manage your term-deposit marketing campaigns."
    )

    st.subheader("Create New Campaign")

    with st.form("campaign_form"):

        campaign_name = st.text_input(
            "Campaign Name"
        )

        target_segment = st.selectbox(
            "Target Customer Segment",
            [
                "Students",
                "Retired",
                "Management",
                "Technicians",
                "Blue-collar",
                "Entrepreneurs",
                "All Customers"
            ]
        )

        budget = st.number_input(
            "Campaign Budget",
            min_value=0.0,
            value=1000.0,
            step=100.0
        )

        start_date = st.date_input(
            "Campaign Start Date"
        )

        notes = st.text_area(
            "Campaign Notes"
        )

        submitted = st.form_submit_button(
            "Save Campaign",
            use_container_width=True
        )

        if submitted:

            if campaign_name.strip():

                add_campaign(
                    campaign_name,
                    target_segment,
                    budget,
                    str(start_date),
                    notes
                )

                st.success("Campaign saved successfully!")

            else:

                st.warning("Please enter a campaign name.")

    st.divider()

    st.subheader("Campaign Overview")

    campaigns = get_campaigns()

    if not campaigns.empty:

        col1, col2, col3 = st.columns(3)

        with col1:
            st.metric(
                "Total Campaigns",
                len(campaigns)
            )

        with col2:
            st.metric(
                "Total Budget",
                f"{campaigns['budget'].sum():,.2f}"
            )

        with col3:
            st.metric(
                "Target Segments",
                campaigns["target_segment"].nunique()
            )

        st.dataframe(
            campaigns,
            use_container_width=True,
            hide_index=True
        )

        st.subheader("Delete Campaign")

        campaign_options = {
            f"{row['campaign_name']} (ID: {row['id']})": row["id"]
            for _, row in campaigns.iterrows()
        }

        selected_campaign = st.selectbox(
            "Select campaign",
            list(campaign_options.keys())
        )

        if st.button("Delete Selected Campaign"):

            delete_campaign(
                campaign_options[selected_campaign]
            )

            st.success("Campaign deleted successfully!")

            st.rerun()

    else:

        st.info("No campaigns created yet. Create your first campaign above.")


# =========================================================
# CUSTOMER INSIGHTS
# =========================================================

elif page == "Customer Insights":

    st.title("👥 Customer Insights")

    st.write(
        "Explore customer segments and their subscription behavior."
    )

    st.subheader("Subscription by Job")

    job_data = (
        df.groupby("job")["y"]
        .apply(lambda x: (x == "yes").mean() * 100)
        .reset_index(name="Subscription Rate")
        .sort_values("Subscription Rate", ascending=False)
    )

    fig = px.bar(
        job_data,
        x="job",
        y="Subscription Rate",
        title="Subscription Rate by Job",
        color="Subscription Rate",
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Customer Age Groups")

    bins = [0, 25, 35, 45, 55, 65, 100]

    labels = [
        "18-25",
        "26-35",
        "36-45",
        "46-55",
        "56-65",
        "65+"
    ]

    age_df = df.copy()

    age_df["Age Group"] = pd.cut(
        age_df["age"],
        bins=bins,
        labels=labels
    )

    age_data = (
        age_df.groupby("Age Group", observed=False)["y"]
        .apply(lambda x: (x == "yes").mean() * 100)
        .reset_index(name="Subscription Rate")
    )

    fig = px.bar(
        age_data,
        x="Age Group",
        y="Subscription Rate",
        title="Subscription Rate by Age Group",
        color="Subscription Rate",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Job Segment Details")

    st.dataframe(
        job_data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# SUBSCRIPTION PREDICTOR
# =========================================================

elif page == "Subscription Predictor":

    st.title("🤖 Subscription Predictor")

    st.write(
        "Predict whether a customer may subscribe to a term deposit "
        "using a Random Forest machine learning model."
    )

    st.warning(
        "Educational prediction only. Predictions are estimates, "
        "not guaranteed customer decisions."
    )

    st.info(
        "The duration feature is excluded because it is only known "
        "after the call and cannot be used for pre-call targeting."
    )

    if st.button("Train Machine Learning Model"):

        with st.spinner("Training Random Forest model..."):

            try:

                model, accuracy, features = train_model()

                st.session_state["model"] = model
                st.session_state["accuracy"] = accuracy
                st.session_state["features"] = features

                st.success("Model trained successfully!")

            except Exception as error:

                st.error(f"Model training failed: {error}")

    if "model" not in st.session_state:

        st.info(
            "Click the Train Machine Learning Model button first "
            "to train the prediction model."
        )

    else:

        model = st.session_state["model"]

        accuracy = st.session_state["accuracy"]

        st.metric(
            "Model Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        st.divider()

        st.subheader("Enter Customer Information")

        features = st.session_state["features"]

        customer = {}

        with st.form("prediction_form"):

            col1, col2 = st.columns(2)

            for index, feature in enumerate(features):

                target_col = col1 if index % 2 == 0 else col2

                with target_col:

                    if feature in df.select_dtypes(
                        include=["number"]
                    ).columns:

                        default_value = float(
                            df[feature].median()
                        )

                        customer[feature] = st.number_input(
                            feature.replace(".", " ").title(),
                            value=default_value
                        )

                    else:

                        options = sorted(
                            df[feature].dropna().astype(str).unique().tolist()
                        )

                        if not options:
                            options = ["unknown"]

                        customer[feature] = st.selectbox(
                            feature.replace(".", " ").title(),
                            options
                        )

            predict_button = st.form_submit_button(
                "Predict Subscription",
                use_container_width=True
            )

        if predict_button:

            try:

                input_df = pd.DataFrame(
                    [customer],
                    columns=features
                )

                prediction = model.predict(input_df)[0]

                probabilities = model.predict_proba(input_df)[0]

                classes = model.classes_

                yes_index = list(classes).index("yes")

                yes_probability = probabilities[yes_index] * 100

                if prediction == "yes":

                    st.success(
                        "Prediction: Customer is likely to subscribe."
                    )

                else:

                    st.warning(
                        "Prediction: Customer is unlikely to subscribe."
                    )

                st.metric(
                    "Estimated Subscription Probability",
                    f"{yes_probability:.2f}%"
                )

            except Exception as error:

                st.error(f"Prediction failed: {error}")


# =========================================================
# MODEL AND DATASET INFORMATION
# =========================================================

elif page == "Model & Dataset":

    st.title("📚 Model & Dataset Information")

    st.write(
        "Information about the UCI Bank Marketing dataset "
        "and the machine learning approach."
    )

    st.subheader("Dataset Information")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Rows",
            f"{df.shape[0]:,}"
        )

    with col2:
        st.metric(
            "Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    st.write(
        "**Target column:** `y` (yes / no)"
    )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.subheader("Dataset Description")

    st.write(
        """
        The Bank Marketing dataset contains information about
        customers contacted during bank marketing campaigns.

        The objective is to understand customer behavior and
        predict whether a customer subscribes to a term deposit.
        """
    )

    st.markdown(
        "[UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)"
    )

    st.divider()

    st.subheader("Machine Learning Model")

    st.write("**Algorithm:** Random Forest Classifier")

    st.write("**Preprocessing:**")

    st.write(
        "- Missing value handling"
    )

    st.write(
        "- One-Hot Encoding for categorical features"
    )

    st.write(
        "- Median imputation for numerical features"
    )

    st.write(
        "- 80% training and 20% testing split"
    )

    st.write(
        "**Target:** Predict term deposit subscription (yes/no)."
    )

    st.write(
        "**Important:** The duration feature is excluded "
        "to avoid data leakage."
    )

    st.divider()

    st.subheader("Project Technologies")

    st.write(
        """
        - Python
        - Streamlit
        - Pandas and NumPy
        - Scikit-learn
        - Plotly
        - SQLite
        """
    )


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Term Deposit Campaign Planner | Academic Project"
)
