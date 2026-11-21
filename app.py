
import streamlit as st
import pandas as pd
import sqlite3
from pathlib import Path

import plotly.express as px

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score


# =========================================================
# PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Term Deposit Campaign Planner",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)


# =========================================================
# COMPLETE DARK THEME
# =========================================================

st.markdown("""
<style>

/* MAIN BACKGROUND */
.stApp {
    background-color: #0B1120 !important;
    color: #F8FAFC !important;
}

[data-testid="stAppViewContainer"] {
    background-color: #0B1120 !important;
}

[data-testid="stHeader"] {
    background-color: #0B1120 !important;
}

/* MAIN CONTENT */
.main .block-container {
    padding-top: 2rem;
    padding-bottom: 2rem;
}

/* HEADINGS */
h1, h2, h3, h4, h5, h6 {
    color: #F8FAFC !important;
    font-weight: 700 !important;
}

/* GENERAL TEXT */
.stApp p,
.stApp li,
.stApp label,
.stApp .stMarkdown {
    color: #DCE5F1 !important;
}

/* SIDEBAR */
section[data-testid="stSidebar"] {
    background-color: #111B2E !important;
    border-right: 1px solid #26354D !important;
}

section[data-testid="stSidebar"] * {
    color: #E2E8F0 !important;
}

/* SIDEBAR RADIO */
section[data-testid="stSidebar"] div[role="radiogroup"] label {
    background-color: transparent !important;
    color: #E2E8F0 !important;
    padding: 9px !important;
    border-radius: 8px !important;
}

section[data-testid="stSidebar"] div[role="radiogroup"] label:hover {
    background-color: #26354D !important;
}

/* METRIC CARDS */
div[data-testid="stMetric"] {
    background-color: #18243A !important;
    border: 1px solid #30415E !important;
    border-radius: 14px !important;
    padding: 20px !important;
    min-height: 115px;
    box-shadow: 0 3px 12px rgba(0,0,0,0.18);
}

/* METRIC LABELS */
div[data-testid="stMetricLabel"],
div[data-testid="stMetricLabel"] p {
    color: #C6D3E3 !important;
    font-size: 14px !important;
    font-weight: 600 !important;
}

/* METRIC VALUES */
div[data-testid="stMetricValue"],
div[data-testid="stMetricValue"] div {
    color: #FFFFFF !important;
    font-size: 29px !important;
    font-weight: 700 !important;
}

/* METRIC DELTA */
div[data-testid="stMetricDelta"],
div[data-testid="stMetricDelta"] p {
    color: #86EFAC !important;
}

/* CONTAINERS */
div[data-testid="stVerticalBlockBorderWrapper"] {
    background-color: #141F33 !important;
    border: 1px solid #2D3D56 !important;
    border-radius: 13px !important;
}

/* INPUTS */
.stTextInput input,
.stNumberInput input,
.stDateInput input,
.stTextArea textarea {
    background-color: #1B2940 !important;
    color: #FFFFFF !important;
    border: 1px solid #42536E !important;
    border-radius: 8px !important;
}

/* INPUT PLACEHOLDERS */
input::placeholder,
textarea::placeholder {
    color: #AAB8CA !important;
}

/* SELECT BOX */
.stSelectbox div[data-baseweb="select"] > div,
.stMultiSelect div[data-baseweb="select"] > div {
    background-color: #1B2940 !important;
    color: #FFFFFF !important;
    border-color: #42536E !important;
}

/* SELECT BOX TEXT */
.stSelectbox div,
.stMultiSelect div {
    color: #FFFFFF !important;
}

/* DROPDOWN MENU */
div[data-baseweb="popover"],
ul[data-baseweb="menu"] {
    background-color: #1B2940 !important;
}

li[data-baseweb="option"] {
    background-color: #1B2940 !important;
    color: #FFFFFF !important;
}

li[data-baseweb="option"]:hover {
    background-color: #30415E !important;
}

/* BUTTONS */
.stButton button,
.stFormSubmitButton button {
    background-color: #2563EB !important;
    color: #FFFFFF !important;
    border: none !important;
    border-radius: 9px !important;
    font-weight: 600 !important;
    padding: 9px 18px !important;
}

/* BUTTON HOVER */
.stButton button:hover,
.stFormSubmitButton button:hover {
    background-color: #1D4ED8 !important;
    color: #FFFFFF !important;
}

/* DATA TABLE */
div[data-testid="stDataFrame"] {
    background-color: #141F33 !important;
    border: 1px solid #30415E !important;
    border-radius: 10px !important;
}

/* CHECKBOX AND RADIO */
.stCheckbox label,
.stRadio label {
    color: #E2E8F0 !important;
}

/* EXPANDERS */
div[data-testid="stExpander"] {
    background-color: #141F33 !important;
    border: 1px solid #30415E !important;
    border-radius: 10px !important;
}

div[data-testid="stExpander"] summary {
    color: #FFFFFF !important;
}

/* TABS */
.stTabs [data-baseweb="tab"] {
    color: #DCE5F1 !important;
}

/* ALERTS */
div[data-testid="stAlert"] {
    background-color: #18243A !important;
    border: 1px solid #30415E !important;
}

div[data-testid="stAlert"] p {
    color: #F1F5F9 !important;
}

/* LINKS */
a {
    color: #60A5FA !important;
}

/* DIVIDER */
hr {
    border-color: #30415E !important;
}

/* CAPTIONS */
.stCaption {
    color: #AAB8CA !important;
}

/* PLOTLY CHART */
div[data-testid="stPlotlyChart"] {
    background-color: #141F33 !important;
    border: 1px solid #30415E !important;
    border-radius: 12px !important;
    padding: 10px !important;
}

/* FILE UPLOADER */
section[data-testid="stFileUploader"] {
    background-color: #18243A !important;
    border-radius: 10px !important;
}

</style>
""", unsafe_allow_html=True)


# =========================================================
# PATH CONFIGURATION
# =========================================================

BASE_DIR = Path(__file__).resolve().parent

DATA_PATH = BASE_DIR / "data" / "bank-additional-full.csv"
ROOT_DATA_PATH = BASE_DIR / "bank-additional-full.csv"

DB_PATH = BASE_DIR / "campaigns.db"


# =========================================================
# DATABASE
# =========================================================

def initialize_database():

    connection = sqlite3.connect(DB_PATH)

    connection.execute("""
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

    connection.execute("""
        INSERT INTO campaigns
        (campaign_name, target_segment, budget, start_date, notes)
        VALUES (?, ?, ?, ?, ?)
    """, (name, segment, budget, start_date, notes))

    connection.commit()
    connection.close()


def get_campaigns():

    connection = sqlite3.connect(DB_PATH)

    result = pd.read_sql_query(
        "SELECT * FROM campaigns ORDER BY id DESC",
        connection
    )

    connection.close()

    return result


def delete_campaign(campaign_id):

    connection = sqlite3.connect(DB_PATH)

    connection.execute(
        "DELETE FROM campaigns WHERE id = ?",
        (int(campaign_id),)
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
        file_path = DATA_PATH

    elif ROOT_DATA_PATH.exists():
        file_path = ROOT_DATA_PATH

    else:
        raise FileNotFoundError(
            "Dataset missing. Upload bank-additional-full.csv "
            "inside the data folder."
        )

    data = pd.read_csv(file_path, sep=";")

    data.columns = data.columns.str.strip()

    if "y" not in data.columns:
        raise ValueError("Target column 'y' is missing.")

    return data


try:
    df = load_dataset()

except Exception as error:

    st.error(f"Dataset could not be loaded: {error}")

    st.info(
        "Required path: data/bank-additional-full.csv"
    )

    st.stop()


# =========================================================
# DATA PREPARATION
# =========================================================

df["y"] = df["y"].astype(str).str.strip().str.lower()

total_customers = len(df)

subscribed = int((df["y"] == "yes").sum())

not_subscribed = int((df["y"] == "no").sum())

subscription_rate = (
    subscribed / total_customers * 100
    if total_customers else 0
)


# =========================================================
# DARK CHART HELPER
# =========================================================

def dark_chart(fig, height=400):

    fig.update_layout(
        template="plotly_dark",
        paper_bgcolor="#141F33",
        plot_bgcolor="#141F33",
        font=dict(
            color="#F1F5F9",
            family="Arial"
        ),
        title_font=dict(
            color="#FFFFFF",
            size=18
        ),
        legend=dict(
            font=dict(color="#E2E8F0")
        ),
        xaxis=dict(
            title_font=dict(color="#E2E8F0"),
            tickfont=dict(color="#E2E8F0"),
            gridcolor="#30415E"
        ),
        yaxis=dict(
            title_font=dict(color="#E2E8F0"),
            tickfont=dict(color="#E2E8F0"),
            gridcolor="#30415E"
        ),
        height=height,
        margin=dict(l=30, r=20, t=65, b=40)
    )

    return fig


# =========================================================
# MACHINE LEARNING MODEL
# =========================================================

@st.cache_resource
def train_model():

    data = load_dataset().copy()

    X = data.drop(
        columns=["y", "duration"],
        errors="ignore"
    )

    y = data["y"].astype(str).str.strip().str.lower()

    numeric_columns = X.select_dtypes(
        include=["number"]
    ).columns.tolist()

    categorical_columns = X.select_dtypes(
        exclude=["number"]
    ).columns.tolist()

    numeric_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="median")
        )
    ])

    categorical_pipeline = Pipeline([
        (
            "imputer",
            SimpleImputer(strategy="most_frequent")
        ),
        (
            "encoder",
            OneHotEncoder(handle_unknown="ignore")
        )
    ])

    preprocessing = ColumnTransformer([
        ("numeric", numeric_pipeline, numeric_columns),
        ("categorical", categorical_pipeline, categorical_columns)
    ])

    classifier = RandomForestClassifier(
        n_estimators=100,
        random_state=42,
        class_weight="balanced",
        n_jobs=-1
    )

    model = Pipeline([
        ("preprocessing", preprocessing),
        ("classifier", classifier)
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

    accuracy = accuracy_score(
        y_test,
        predictions
    )

    return model, accuracy, X.columns.tolist()


# =========================================================
# SIDEBAR
# =========================================================

st.sidebar.title("📊 Term Deposit Planner")

st.sidebar.caption("Marketing Campaign Management")

st.sidebar.divider()

page = st.sidebar.radio(
    "NAVIGATION",
    [
        "Dashboard",
        "Campaign Planner",
        "Customer Insights",
        "Subscription Predictor",
        "Model & Dataset"
    ]
)

st.sidebar.divider()

st.sidebar.caption("Machine Learning Academic Project")

st.sidebar.caption("Powered by Streamlit")


# =========================================================
# PAGE 1: DASHBOARD
# =========================================================

if page == "Dashboard":

    st.title("📊 Dashboard")

    st.markdown(
        "Welcome to the **Term Deposit Campaign Planner**. "
        "Monitor customer subscriptions and marketing performance."
    )

    st.divider()

    st.subheader("Campaign Overview")

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric(
            "Total Customers",
            f"{total_customers:,}"
        )

    with col2:
        st.metric(
            "Subscribed",
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

    st.subheader("Subscription Analytics")

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

        st.plotly_chart(
            dark_chart(fig),
            use_container_width=True
        )

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

        st.plotly_chart(
            dark_chart(fig),
            use_container_width=True
        )

    st.subheader("Subscription Distribution")

    distribution = pd.DataFrame({
        "Subscription": ["Yes", "No"],
        "Customers": [subscribed, not_subscribed]
    })

    fig = px.pie(
        distribution,
        names="Subscription",
        values="Customers",
        hole=0.5,
        title="Customer Subscription Distribution",
        color_discrete_sequence=["#22C55E", "#64748B"]
    )

    st.plotly_chart(
        dark_chart(fig),
        use_container_width=True
    )


# =========================================================
# PAGE 2: CAMPAIGN PLANNER
# =========================================================

elif page == "Campaign Planner":

    st.title("📋 Campaign Planner")

    st.write(
        "Create, monitor and manage your marketing campaigns."
    )

    st.divider()

    st.subheader("Create New Campaign")

    with st.form("campaign_form"):

        campaign_name = st.text_input(
            "Campaign Name",
            placeholder="Enter campaign name"
        )

        target_segment = st.selectbox(
            "Target Customer Segment",
            [
                "All Customers",
                "Students",
                "Retired",
                "Management",
                "Technicians",
                "Blue-collar",
                "Entrepreneurs"
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
            "Campaign Description / Notes",
            placeholder="Enter campaign details..."
        )

        submit = st.form_submit_button(
            "➕ Create Campaign",
            use_container_width=True
        )

        if submit:

            if campaign_name.strip():

                add_campaign(
                    campaign_name.strip(),
                    target_segment,
                    budget,
                    str(start_date),
                    notes
                )

                st.success(
                    "Campaign created successfully!"
                )

            else:

                st.warning(
                    "Please enter a campaign name."
                )

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
                "Customer Segments",
                campaigns["target_segment"].nunique()
            )

        st.divider()

        st.subheader("Saved Campaigns")

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

        selected = st.selectbox(
            "Select campaign to delete",
            list(campaign_options.keys())
        )

        if st.button("🗑️ Delete Selected Campaign"):

            delete_campaign(campaign_options[selected])

            st.success("Campaign deleted successfully!")

            st.rerun()

    else:

        st.info(
            "No campaigns available. Create your first campaign above."
        )


# =========================================================
# PAGE 3: CUSTOMER INSIGHTS
# =========================================================

elif page == "Customer Insights":

    st.title("👥 Customer Insights")

    st.write(
        "Understand customer segments and subscription behavior."
    )

    st.divider()

    st.subheader("Customer Subscription by Job")

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
        color="Subscription Rate",
        title="Subscription Rate by Job",
        color_continuous_scale="Viridis"
    )

    st.plotly_chart(
        dark_chart(fig),
        use_container_width=True
    )

    st.divider()

    st.subheader("Customer Age Group Analysis")

    age_df = df.copy()

    age_df["Age Group"] = pd.cut(
        age_df["age"],
        bins=[0, 25, 35, 45, 55, 65, 100],
        labels=[
            "18-25",
            "26-35",
            "36-45",
            "46-55",
            "56-65",
            "65+"
        ]
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
        color="Subscription Rate",
        title="Subscription Rate by Age Group",
        color_continuous_scale="Blues"
    )

    st.plotly_chart(
        dark_chart(fig),
        use_container_width=True
    )

    st.divider()

    st.subheader("Job Segment Details")

    st.dataframe(
        job_data,
        use_container_width=True,
        hide_index=True
    )


# =========================================================
# PAGE 4: SUBSCRIPTION PREDICTOR
# =========================================================

elif page == "Subscription Predictor":

    st.title("🤖 Subscription Predictor")

    st.write(
        "Use Machine Learning to estimate whether a customer "
        "may subscribe to a term deposit."
    )

    st.warning(
        "Predictions are educational estimates and are not guaranteed."
    )

    st.info(
        "The duration feature is excluded because call duration "
        "is only known after the call."
    )

    if st.button("🚀 Train Random Forest Model"):

        with st.spinner("Training model. Please wait..."):

            try:

                model, accuracy, features = train_model()

                st.session_state["model"] = model
                st.session_state["accuracy"] = accuracy
                st.session_state["features"] = features

                st.success("Model trained successfully!")

            except Exception as error:

                st.error(f"Model training failed: {error}")

    if "model" in st.session_state:

        model = st.session_state["model"]

        accuracy = st.session_state["accuracy"]

        features = st.session_state["features"]

        st.metric(
            "Model Test Accuracy",
            f"{accuracy * 100:.2f}%"
        )

        st.divider()

        st.subheader("Enter Customer Details")

        customer = {}

        with st.form("prediction_form"):

            col1, col2 = st.columns(2)

            for index, feature in enumerate(features):

                target_col = col1 if index % 2 == 0 else col2

                with target_col:

                    if feature in df.select_dtypes(
                        include=["number"]
                    ).columns:

                        value = float(df[feature].median())

                        customer[feature] = st.number_input(
                            feature.replace(".", " ").title(),
                            value=value
                        )

                    else:

                        options = sorted(
                            df[feature].dropna().astype(str).unique()
                        )

                        customer[feature] = st.selectbox(
                            feature.replace(".", " ").title(),
                            options
                        )

            predict = st.form_submit_button(
                "🔍 Predict Subscription",
                use_container_width=True
            )

        if predict:

            try:

                input_data = pd.DataFrame(
                    [customer],
                    columns=features
                )

                prediction = model.predict(input_data)[0]

                probabilities = model.predict_proba(input_data)[0]

                classes = list(model.classes_)

                yes_probability = (
                    probabilities[classes.index("yes")] * 100
                )

                st.divider()

                if prediction == "yes":

                    st.success(
                        "Customer is predicted to subscribe."
                    )

                else:

                    st.warning(
                        "Customer is predicted not to subscribe."
                    )

                st.metric(
                    "Estimated Subscription Probability",
                    f"{yes_probability:.2f}%"
                )

                st.progress(
                    float(yes_probability / 100)
                )

            except Exception as error:

                st.error(f"Prediction error: {error}")

    else:

        st.info(
            "First click 'Train Random Forest Model' to start prediction."
        )


# =========================================================
# PAGE 5: MODEL AND DATASET
# =========================================================

elif page == "Model & Dataset":

    st.title("📚 Model & Dataset Information")

    st.write(
        "Explore the dataset and machine learning methodology "
        "used in this academic project."
    )

    st.divider()

    st.subheader("Dataset Overview")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric(
            "Total Rows",
            f"{df.shape[0]:,}"
        )

    with col2:
        st.metric(
            "Total Columns",
            df.shape[1]
        )

    with col3:
        st.metric(
            "Missing Values",
            int(df.isnull().sum().sum())
        )

    st.divider()

    st.subheader("Dataset Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True,
        hide_index=True
    )

    st.divider()

    st.subheader("Dataset Description")

    st.write(
        """
        The UCI Bank Marketing dataset contains customer information
        and marketing campaign details collected by a bank.

        The objective is to analyze customer behavior and predict
        whether a customer subscribes to a term deposit.
        """
    )

    st.markdown(
        "[UCI Bank Marketing Dataset](https://archive.ics.uci.edu/dataset/222/bank+marketing)"
    )

    st.divider()

    st.subheader("Machine Learning Methodology")

    st.write("**Algorithm:** Random Forest Classifier")

    st.write("**Data Preprocessing:**")

    st.write("- Numerical missing value imputation using median.")
    st.write("- Categorical missing value imputation using mode.")
    st.write("- One-Hot Encoding for categorical columns.")
    st.write("- 80% training data and 20% testing data.")

    st.write("**Prediction Target:**")

    st.write("Term deposit subscription (Yes / No).")

    st.write(
        "**Data Leakage Prevention:** The duration column is "
        "excluded because it is only available after a call."
    )

    st.divider()

    st.subheader("Technologies Used")

    st.write("""
    - Python
    - Streamlit
    - Pandas
    - NumPy
    - Scikit-learn
    - Plotly
    - SQLite
    """)


# =========================================================
# FOOTER
# =========================================================

st.sidebar.divider()

st.sidebar.caption(
    "Term Deposit Campaign Planner"
)

st.sidebar.caption(
    "Academic Machine Learning Project"
)
