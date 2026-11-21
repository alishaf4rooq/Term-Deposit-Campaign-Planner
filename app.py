import os
import sqlite3
from datetime import date

import pandas as pd
import plotly.express as px
import streamlit as st

from database import init_db, add_campaign, get_campaigns, delete_campaign
from model_utils import load_dataset, train_model, predict_subscription, FEATURE_COLUMNS, TARGET

st.set_page_config(
    page_title="Term-Deposit Campaign Planner",
    page_icon="🏦",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown("""
<style>
    .main {background-color: #f7f9fc;}
    [data-testid="stSidebar"] {background: #10243a;}
    [data-testid="stSidebar"] * {color: #ffffff;}
    .hero {
        padding: 1.35rem 1.6rem; border-radius: 16px;
        background: linear-gradient(115deg, #10243a, #176b87);
        color: white; margin-bottom: 1rem;
    }
    .hero h1 {color: white; margin-bottom: .25rem;}
    .hero p {color: #e5f4fa; margin: 0;}
    div[data-testid="stMetric"] {
        background: white; padding: 14px; border-radius: 12px;
        border: 1px solid #e7edf3;
    }
    .small-note {color: #617184; font-size: .9rem;}
</style>
""", unsafe_allow_html=True)

DATA_PATH = os.path.join("data", "bank-additional-full.csv")
init_db()

@st.cache_data
def get_data():
    return load_dataset(DATA_PATH)

@st.cache_resource
def get_trained_model():
    data = get_data()
    return train_model(data)

try:
    df = get_data()
except Exception as exc:
    st.error(f"Dataset could not be loaded: {exc}")
    st.info("Check that data/bank-additional-full.csv exists and is the original semicolon-separated CSV.")
    st.stop()

with st.sidebar:
    st.title("🏦 TD Campaign")
    st.caption("Bank Marketing Analytics")
    page = st.radio(
        "NAVIGATION",
        ["Dashboard", "Campaign Planner", "Customer Insights", "Subscription Predictor", "Model & Dataset"],
    )
    st.divider()
    st.caption("Final Year / Academic Project")
    st.caption("Streamlit • Python • ML • SQLite")

st.markdown("""
<div class="hero">
  <h1>Term-Deposit Campaign Planner</h1>
  <p>Explore bank marketing data, plan outreach campaigns, and estimate subscription likelihood.</p>
</div>
""", unsafe_allow_html=True)

yes_count = int((df[TARGET] == "yes").sum())
no_count = int((df[TARGET] == "no").sum())
total = len(df)
rate = yes_count / total * 100

if page == "Dashboard":
    st.subheader("Campaign Overview")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Total Customers", f"{total:,}")
    c2.metric("Subscribed", f"{yes_count:,}")
    c3.metric("Not Subscribed", f"{no_count:,}")
    c4.metric("Subscription Rate", f"{rate:.2f}%")

    left, right = st.columns(2)
    with left:
        channel = df.groupby("contact", as_index=False)[TARGET].apply(
            lambda s: (s == "yes").mean() * 100
        )
        channel.columns = ["Contact Channel", "Subscription Rate"]
        fig = px.bar(channel, x="Contact Channel", y="Subscription Rate",
                     title="Subscription Rate by Contact Channel", text_auto=".1f")
        st.plotly_chart(fig, use_container_width=True)
    with right:
        month = df.groupby("month", as_index=False)[TARGET].apply(
            lambda s: (s == "yes").mean() * 100
        )
        month.columns = ["Month", "Subscription Rate"]
        order = ["mar", "apr", "may", "jun", "jul", "aug", "sep", "oct", "nov", "dec"]
        month["Month"] = pd.Categorical(month["Month"], categories=order, ordered=True)
        month = month.sort_values("Month")
        fig = px.line(month, x="Month", y="Subscription Rate",
                      title="Subscription Rate by Month", markers=True)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Dataset Preview")
    st.dataframe(df.head(10), use_container_width=True, hide_index=True)
    st.caption("The supplied UCI Bank Marketing additional dataset contains 41,188 customer records.")

elif page == "Campaign Planner":
    st.subheader("Create and Manage Campaign Plans")
    st.write("Save an outreach plan in the local SQLite database.")
    with st.form("campaign_form", clear_on_submit=True):
        a, b = st.columns(2)
        with a:
            campaign_name = st.text_input("Campaign name", placeholder="e.g. Spring Term Deposit Drive")
            target_segment = st.selectbox("Target customer segment",
                ["All customers", "Retired", "Students / young adults", "Management", "Technicians",
                 "Admin", "Blue-collar", "Services", "Entrepreneurs", "Other"])
            channel_choice = st.selectbox("Contact channel", ["Cellular", "Telephone", "Both"])
        with b:
            start_date = st.date_input("Planned start date", value=date.today())
            budget = st.number_input("Estimated budget", min_value=0.0, step=100.0)
            target_count = st.number_input("Target customers", min_value=1, max_value=100000, value=500)
        objective = st.text_area("Campaign objective", placeholder="Describe the goal of this campaign...")
        submitted = st.form_submit_button("Save Campaign", type="primary")
        if submitted:
            if not campaign_name.strip():
                st.error("Please enter a campaign name.")
            else:
                add_campaign(campaign_name.strip(), target_segment, channel_choice,
                             str(start_date), float(budget), int(target_count), objective.strip())
                st.success("Campaign saved successfully.")

    st.divider()
    st.subheader("Saved Campaigns")
    campaigns = get_campaigns()
    if campaigns.empty:
        st.info("No campaign plans saved yet. Create your first plan above.")
    else:
        st.dataframe(campaigns, use_container_width=True, hide_index=True)
        ids = campaigns["id"].tolist()
        selected_id = st.selectbox("Select campaign ID to delete", ids)
        if st.button("Delete Selected Campaign"):
            delete_campaign(int(selected_id))
            st.success("Campaign deleted.")
            st.rerun()

elif page == "Customer Insights":
    st.subheader("Explore Customer Segments")
    selected_job = st.selectbox("Filter by job", ["All"] + sorted(df["job"].dropna().unique().tolist()))
    view = df if selected_job == "All" else df[df["job"] == selected_job]
    a, b, c = st.columns(3)
    a.metric("Customers in selection", f"{len(view):,}")
    b.metric("Subscriptions", f"{(view[TARGET] == 'yes').sum():,}")
    b_rate = (view[TARGET] == "yes").mean() * 100 if len(view) else 0
    c.metric("Subscription rate", f"{b_rate:.2f}%")

    job_rates = (df.groupby("job")[TARGET]
                   .apply(lambda s: (s == "yes").mean() * 100)
                   .reset_index(name="subscription_rate")
                   .sort_values("subscription_rate", ascending=False))
    fig = px.bar(job_rates, x="job", y="subscription_rate",
                 title="Subscription Rate by Job", text_auto=".1f")
    st.plotly_chart(fig, use_container_width=True)

    age_view = df.copy()
    age_view["age_group"] = pd.cut(age_view["age"], bins=[0, 25, 35, 45, 55, 65, 100],
                                   labels=["18–25", "26–35", "36–45", "46–55", "56–65", "65+"])
    age_rates = (age_view.groupby("age_group", observed=False)[TARGET]
                 .apply(lambda s: (s == "yes").mean() * 100)
                 .reset_index(name="subscription_rate"))
    fig = px.bar(age_rates, x="age_group", y="subscription_rate",
                 title="Subscription Rate by Age Group", text_auto=".1f")
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Filtered Customer Records")
    st.dataframe(view.head(100), use_container_width=True, hide_index=True)
    st.caption("Only the first 100 filtered records are displayed.")

elif page == "Subscription Predictor":
    st.subheader("Predict Term-Deposit Subscription")
    st.warning(
        "This is an educational prediction, not a guarantee. The model excludes call duration "
        "because duration is only known after a call and would cause information leakage for pre-call targeting."
    )
    with st.spinner("Preparing the machine-learning model (first run may take a little time)..."):
        try:
            model, metrics = get_trained_model()
        except Exception as exc:
            st.error(f"Model training failed: {exc}")
            st.stop()

    st.success("Model is ready.")
    with st.expander("Model evaluation", expanded=False):
        m1, m2, m3 = st.columns(3)
        m1.metric("Test Accuracy", f"{metrics['accuracy']*100:.1f}%")
        m2.metric("Precision (yes)", f"{metrics['precision']*100:.1f}%")
        m3.metric("Recall (yes)", f"{metrics['recall']*100:.1f}%")
        st.caption("Metrics are calculated on a held-out test split. Accuracy alone can be misleading because the classes are imbalanced.")

    defaults = {
        "age": 40, "job": "admin.", "marital": "married", "education": "university.degree",
        "default": "no", "housing": "yes", "loan": "no", "contact": "cellular",
        "month": "may", "day_of_week": "thu", "campaign": 1, "pdays": 999, "previous": 0,
        "poutcome": "nonexistent", "emp.var.rate": 1.1, "cons.price.idx": 93.994,
        "cons.conf.idx": -36.4, "euribor3m": 4.857, "nr.employed": 5191.0
    }
    numeric_cols = ["age", "campaign", "pdays", "previous", "emp.var.rate",
                    "cons.price.idx", "cons.conf.idx", "euribor3m", "nr.employed"]
    categorical_cols = [c for c in FEATURE_COLUMNS if c not in numeric_cols]

    with st.form("prediction_form"):
        st.markdown("**Customer profile**")
        cols = st.columns(3)
        values = {}
        for i, col in enumerate(numeric_cols):
            with cols[i % 3]:
                if col == "age":
                    values[col] = st.number_input("Age", 18, 100, int(defaults[col]))
                elif col == "campaign":
                    values[col] = st.number_input("Contacts in this campaign", 1, 100, int(defaults[col]))
                elif col == "pdays":
                    values[col] = st.number_input("Days since previous contact (999 = not contacted)", 0, 999, int(defaults[col]))
                elif col == "previous":
                    values[col] = st.number_input("Previous contacts", 0, 100, int(defaults[col]))
                else:
                    values[col] = st.number_input(col, value=float(defaults[col]), format="%.3f")
        st.markdown("**Customer and campaign details**")
        cols = st.columns(3)
        for i, col in enumerate(categorical_cols):
            choices = sorted(df[col].dropna().astype(str).unique().tolist())
            default = defaults.get(col, choices[0])
            if default not in choices:
                default = choices[0]
            with cols[i % 3]:
                values[col] = st.selectbox(col.replace("_", " ").title(), choices,
                                           index=choices.index(default))
        run_prediction = st.form_submit_button("Predict Subscription Likelihood", type="primary")

    if run_prediction:
        result = predict_subscription(model, values)
        probability = result["probability"]
        st.divider()
        if result["prediction"] == "yes":
            st.success("Model prediction: Customer is more likely to subscribe.")
        else:
            st.info("Model prediction: Customer is less likely to subscribe.")
        st.metric("Estimated subscription probability", f"{probability*100:.1f}%")
        st.progress(float(probability))
        st.caption("Probability is a model estimate and should support—not replace—human campaign decisions.")

elif page == "Model & Dataset":
    st.subheader("Dataset Information")
    info1, info2, info3 = st.columns(3)
    info1.metric("Rows", f"{len(df):,}")
    info2.metric("Columns", f"{df.shape[1]}")
    info3.metric("Missing values", f"{int(df.isna().sum().sum()):,}")
    st.write("**Target column:** `y` (yes / no)")
    st.write("**File:** `data/bank-additional-full.csv`")
    st.write("**Separator:** semicolon (`;`)")
    st.write("**Dataset source:** UCI Machine Learning Repository — Bank Marketing")
    st.link_button("Open UCI dataset page", "https://archive.ics.uci.edu/dataset/222/bank+marketing")

    st.subheader("Machine-Learning Approach")
    st.markdown("""
    1. Load the supplied CSV using the semicolon separator.
    2. Separate input features from target `y`.
    3. Exclude `duration` from prediction features to avoid post-call information leakage.
    4. Impute missing values, one-hot encode categorical variables, and scale numeric variables.
    5. Train a Random Forest classifier and evaluate it on a held-out test set.
    """)
    st.write("**Input features used:**")
    st.code(", ".join(FEATURE_COLUMNS))
    st.write("**Target:** `y`")
    st.caption("The dataset contains historical marketing outcomes. Predictions are estimates and may not generalize to every bank or future campaign.")
