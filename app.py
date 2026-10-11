from pathlib import Path

import joblib
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ROOT = Path(__file__).parent
MODEL_PATH = ROOT / "models" / "churn_model_bundle.joblib"
DATA_PATH = ROOT / "data" / "telco_customer_churn.csv"
PAGES = ["Overview", "Predict Customer", "Customer Insights", "Model Lab"]

st.set_page_config(
    page_title="ChurnSense AI | Customer Churn",
    page_icon="◈",
    layout="wide",
    initial_sidebar_state="expanded",
)

st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Manrope:wght@500;600;700;800&display=swap');
    :root { --ink:#edf5f5; --muted:#91a9b2; --panel:#101f2b; --line:#203642;
            --teal:#36d6c2; --teal-dim:#173f44; --canvas:#08131d; }
    html, body, [class*="css"] { font-family:'DM Sans', sans-serif; }
    .stApp { background: radial-gradient(ellipse at 85% 0%, #12313a 0%, transparent 38%), var(--canvas); color:var(--ink); }
    [data-testid="stHeader"] { background:rgba(8,19,29,.75); }
    [data-testid="stSidebar"] { background:#0b1923; border-right:1px solid var(--line); }
    [data-testid="stSidebar"] [data-testid="stMarkdown"] p { color:var(--muted); }
    h1,h2,h3 { font-family:'Manrope', sans-serif !important; color:var(--ink) !important; letter-spacing:0 !important; }
    h1 { font-size:2.25rem !important; font-weight:800 !important; }
    h2 { font-size:1.45rem !important; font-weight:700 !important; }
    h3 { font-size:1.05rem !important; font-weight:700 !important; }
    p, label, li { color:#c5d3d6; }
    .brand { font:800 1.22rem 'Manrope',sans-serif; color:#f2fbfb; letter-spacing:0; }
    .brand span { color:var(--teal); }
    .eyebrow { color:var(--teal); text-transform:uppercase; font-size:.72rem; font-weight:700; letter-spacing:.12em; }
    .tagline { font-size:1.02rem; color:#b8cacc; margin-top:-.35rem; }
    .subtle { color:var(--muted); font-size:.9rem; }
    .panel { background:linear-gradient(145deg,rgba(18,37,49,.96),rgba(13,28,39,.96)); border:1px solid var(--line); border-radius:14px; padding:1.15rem 1.25rem; }
    .callout { background:rgba(54,214,194,.08); border:1px solid rgba(54,214,194,.28); border-left:3px solid var(--teal); border-radius:8px; padding:.9rem 1rem; color:#cce9e7; }
    [data-testid="stMetric"] { background:linear-gradient(145deg,#122633,#0e1d28); border:1px solid var(--line); border-radius:12px; padding:1rem 1rem .85rem; }
    [data-testid="stMetricLabel"] { color:var(--muted) !important; }
    [data-testid="stMetricValue"] { color:#f2fbfb !important; font-family:'Manrope',sans-serif; }
    [data-testid="stMetricDelta"] { color:var(--teal) !important; }
    .stButton button, .stFormSubmitButton button { background:var(--teal); color:#062126; border:0; border-radius:8px; font-weight:700; min-height:2.7rem; }
    .stButton button:hover, .stFormSubmitButton button:hover { background:#66e7d7; color:#062126; border:0; }
    .stButton button:disabled, .stFormSubmitButton button:disabled { background:#28404a; color:#849ba1; }
    [data-baseweb="select"] > div, [data-testid="stNumberInput"] input, [data-testid="stTextInput"] input { background:#0c1c27; border-color:#29414d; }
    [data-testid="stSlider"] [role="slider"] { background:var(--teal); }
    div[data-testid="stTabs"] button { color:#9fb4bb; }
    div[data-testid="stTabs"] button[aria-selected="true"] { color:var(--teal); }
    hr { border-color:var(--line); }
    @media (max-width:700px) { h1 {font-size:1.75rem !important;} .panel {padding:.9rem;} }
    </style>
    """,
    unsafe_allow_html=True,
)


def load_bundle(path, modified_time):
    """Load the student's trusted, locally-created joblib bundle."""
    del modified_time  # Included in the cache key so replacing the bundle reloads it.
    bundle = joblib.load(path)
    if not isinstance(bundle, dict) or "model" not in bundle:
        raise ValueError("Bundle must be a dictionary containing a fitted model under the 'model' key.")
    return bundle


def load_dataset(uploaded_file):
    if uploaded_file is not None:
        return pd.read_csv(uploaded_file), "Uploaded CSV"
    if DATA_PATH.exists():
        return pd.read_csv(DATA_PATH), str(DATA_PATH.relative_to(ROOT))
    return None, None


def clean_dataset(frame):
    frame = frame.copy()
    frame.columns = [str(column).strip() for column in frame.columns]
    required = {"Churn", "Contract", "tenure", "MonthlyCharges"}
    missing = sorted(required.difference(frame.columns))
    if missing:
        raise ValueError("The CSV is missing required IBM Telco columns: " + ", ".join(missing))
    frame["Churn"] = frame["Churn"].astype(str).str.strip()
    frame["Contract"] = frame["Contract"].astype(str).str.strip()
    frame["tenure"] = pd.to_numeric(frame["tenure"], errors="coerce")
    frame["MonthlyCharges"] = pd.to_numeric(frame["MonthlyCharges"], errors="coerce")
    if "TotalCharges" in frame:
        frame["TotalCharges"] = pd.to_numeric(frame["TotalCharges"], errors="coerce")
    return frame


def chart_layout(fig, height=330):
    fig.update_layout(
        template="plotly_dark",
        height=height,
        margin=dict(l=12, r=12, t=28, b=12),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        font=dict(family="DM Sans, sans-serif", color="#c5d3d6"),
        legend_title_text="",
    )
    fig.update_xaxes(gridcolor="#203642", zerolinecolor="#203642")
    fig.update_yaxes(gridcolor="#203642", zerolinecolor="#203642")
    return fig


def metric_value(metrics, key):
    value = metrics.get(key)
    if value is None:
        return "Not provided"
    try:
        number = float(value)
    except (TypeError, ValueError):
        return str(value)
    return f"{number:.1%}" if 0 <= number <= 1 else f"{number:.2f}"


def feature_importances(bundle):
    supplied = bundle.get("feature_importances")
    if isinstance(supplied, dict) and supplied:
        return pd.DataFrame({"Feature": list(supplied), "Importance": list(supplied.values())})
    model = bundle["model"]
    values = getattr(model, "feature_importances_", None)
    if values is None:
        return None
    names = bundle.get("feature_columns")
    if names is None:
        names = getattr(model, "feature_names_in_", None)
    if names is None or len(names) != len(values):
        return None
    return pd.DataFrame({"Feature": list(names), "Importance": values})


def raw_customer(values):
    return pd.DataFrame([values])


def prepare_model_input(customer, bundle):
    model = bundle["model"]
    mode = bundle.get("input_mode")
    columns = bundle.get("feature_columns")
    if mode is None:
        model_columns = getattr(model, "feature_names_in_", None)
        mode = "raw" if model_columns is not None and set(model_columns).issubset(customer.columns) else "encoded"

    if mode == "raw":
        if columns is None:
            columns = getattr(model, "feature_names_in_", customer.columns)
        missing = sorted(set(columns).difference(customer.columns))
        if missing:
            raise ValueError("Raw model input expects fields not present in the form: " + ", ".join(missing))
        return customer.loc[:, list(columns)]

    if mode == "encoded":
        if columns is None:
            columns = getattr(model, "feature_names_in_", None)
        if columns is None:
            raise ValueError("Encoded bundles must include 'feature_columns' in the model bundle.")
        encoded = pd.get_dummies(customer)
        return encoded.reindex(columns=list(columns), fill_value=0)

    raise ValueError("'input_mode' must be either 'raw' or 'encoded'.")


def positive_class_index(model):
    classes = list(getattr(model, "classes_", []))
    if len(classes) < 2:
        raise ValueError("The fitted model does not expose two churn classes.")
    for index, label in enumerate(classes):
        if str(label).strip().lower() in {"yes", "1", "true", "churn", "churned"}:
            return index
    return 1


def make_prediction(customer, bundle):
    model = bundle["model"]
    if not hasattr(model, "predict_proba"):
        raise ValueError("This model has no predict_proba method, so it cannot provide the requested probability.")
    model_input = prepare_model_input(customer, bundle)
    probabilities = model.predict_proba(model_input)[0]
    churn_probability = float(probabilities[positive_class_index(model)])
    if not 0 <= churn_probability <= 1:
        raise ValueError("The model returned a probability outside the expected 0–1 range.")
    return churn_probability


def render_overview(data):
    st.markdown('<div class="eyebrow">Customer retention intelligence</div>', unsafe_allow_html=True)
    st.title("See risk before it becomes churn.")
    st.markdown('<div class="tagline">See the risk. Understand the why. Retain the relationship.</div>', unsafe_allow_html=True)
    st.write("")
    st.markdown(
        '<div class="panel"><h3>ChurnSense AI</h3><p>Explore customer churn patterns and, once your trained model bundle is installed, assess an individual customer profile. Use the signals to guide a thoughtful retention conversation, not as a guarantee of what a customer will do.</p></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    st.subheader("Portfolio snapshot")
    if data is None:
        st.info("No customer dataset is loaded. Upload the IBM Telco Customer Churn CSV in the sidebar, or place it at `data/telco_customer_churn.csv`, to populate portfolio metrics and charts.")
        values = [("Customers", "—"), ("Churn rate", "—"), ("Avg. monthly charges", "—"), ("Avg. tenure", "—")]
    else:
        count = len(data)
        churned = int(data["Churn"].str.lower().eq("yes").sum())
        values = [
            ("Customers", f"{count:,}"),
            ("Churn rate", f"{churned / count:.1%}" if count else "—"),
            ("Avg. monthly charges", f"${data['MonthlyCharges'].mean():,.2f}"),
            ("Avg. tenure", f"{data['tenure'].mean():.1f} mo"),
        ]
    cards = st.columns(4)
    for card, (label, value) in zip(cards, values):
        card.metric(label, value)

    st.write("")
    left, right = st.columns([1.05, 1.2], gap="large")
    with left:
        st.subheader("Churn at a glance")
        if data is not None:
            counts = data["Churn"].str.title().value_counts().reindex(["No", "Yes"], fill_value=0).rename_axis("Outcome").reset_index(name="Customers")
            fig = px.pie(counts, names="Outcome", values="Customers", hole=.68, color="Outcome", color_discrete_map={"No": "#36d6c2", "Yes": "#ff8a72"})
            fig.update_traces(textposition="inside", textinfo="percent", marker=dict(line=dict(color="#0b1923", width=3)))
            st.plotly_chart(chart_layout(fig, 300), use_container_width=True)
        else:
            st.markdown('<div class="panel"><p class="subtle">Your dataset will populate this view with observed customer counts. No sample values are substituted.</p></div>', unsafe_allow_html=True)
    with right:
        st.subheader("A signal, not a certainty")
        st.markdown(
            '<div class="panel"><p><b>1. Observe</b><br>Review customer patterns in the IBM Telco data.</p><p><b>2. Assess</b><br>Submit a profile to the saved Decision Tree when the model bundle is available.</p><p><b>3. Respond</b><br>Use the risk signal to inform a human-led retention action.</p><hr><p class="subtle">Model outputs are estimates from a student project. They are not guarantees or a substitute for customer context.</p></div>',
            unsafe_allow_html=True,
        )


def render_prediction(bundle):
    st.markdown('<div class="eyebrow">Individual customer assessment</div>', unsafe_allow_html=True)
    st.title("Predict Customer")
    if bundle is None:
        st.warning(f"Prediction is unavailable because the trained model bundle was not found at `{MODEL_PATH.relative_to(ROOT)}`. Add your fitted bundle there after exporting it from Colab; this app will not generate substitute predictions.")

    with st.form("customer_form"):
        st.subheader("Customer profile")
        a, b, c = st.columns(3)
        with a:
            gender = st.selectbox("Gender", ["Female", "Male"])
            senior = st.selectbox("Senior citizen", ["No", "Yes"])
            partner = st.selectbox("Has partner", ["Yes", "No"])
            dependents = st.selectbox("Has dependents", ["No", "Yes"])
            tenure = st.slider("Tenure (months)", 0, 72, 12)
        with b:
            phone_service = st.selectbox("Phone service", ["Yes", "No"])
            multiple_lines = st.selectbox("Multiple lines", ["No", "Yes", "No phone service"])
            internet = st.selectbox("Internet service", ["Fiber optic", "DSL", "No"])
            online_security = st.selectbox("Online security", ["No", "Yes", "No internet service"])
            online_backup = st.selectbox("Online backup", ["No", "Yes", "No internet service"])
            device_protection = st.selectbox("Device protection", ["No", "Yes", "No internet service"])
        with c:
            tech_support = st.selectbox("Tech support", ["No", "Yes", "No internet service"])
            streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
            streaming_movies = st.selectbox("Streaming movies", ["No", "Yes", "No internet service"])
            contract = st.selectbox("Contract", ["Month-to-month", "One year", "Two year"])
            paperless = st.selectbox("Paperless billing", ["Yes", "No"])
            payment = st.selectbox("Payment method", ["Electronic check", "Mailed check", "Bank transfer (automatic)", "Credit card (automatic)"])
        d, e = st.columns(2)
        with d:
            monthly = st.number_input("Monthly charges ($)", min_value=0.0, max_value=1000.0, value=70.0, step=1.0)
        with e:
            total_default = float(tenure) * monthly
            total = st.number_input("Total charges ($)", min_value=0.0, max_value=100000.0, value=total_default, step=10.0)
        st.caption("Total charges are prefilled as tenure × monthly charges; adjust the value if you have the customer's actual amount.")
        submitted = st.form_submit_button("Analyze Customer Risk", disabled=bundle is None, use_container_width=True)

    if submitted and bundle is not None:
        customer = raw_customer({
            "gender": gender,
            "SeniorCitizen": 1 if senior == "Yes" else 0,
            "Partner": partner,
            "Dependents": dependents,
            "tenure": tenure,
            "PhoneService": phone_service,
            "MultipleLines": multiple_lines,
            "InternetService": internet,
            "OnlineSecurity": online_security,
            "OnlineBackup": online_backup,
            "DeviceProtection": device_protection,
            "TechSupport": tech_support,
            "StreamingTV": streaming_tv,
            "StreamingMovies": streaming_movies,
            "Contract": contract,
            "PaperlessBilling": paperless,
            "PaymentMethod": payment,
            "MonthlyCharges": monthly,
            "TotalCharges": total,
        })
        try:
            probability = make_prediction(customer, bundle)
            is_churn_risk = probability >= .5
            st.write("")
            result, details = st.columns([1, 1.25])
            with result:
                if is_churn_risk:
                    st.error("Likely to leave")
                else:
                    st.success("Likely to stay")
                st.metric("Estimated churn probability", f"{probability:.1%}")
            with details:
                st.markdown('<div class="callout"><b>How to use this signal</b><br>Review the customer context and consider a relevant follow-up. This score reflects the saved model and input profile; it is not a real-world guarantee.</div>', unsafe_allow_html=True)
        except Exception as error:
            st.error(f"Could not analyze this profile: {error}")
            st.caption("Check that the bundle input mode, feature columns, and training-time preprocessing match the IBM Telco dataset and this form.")


def render_insights(data, bundle):
    st.markdown('<div class="eyebrow">Patterns behind retention</div>', unsafe_allow_html=True)
    st.title("Customer Insights")
    if data is None:
        st.info("Upload a CSV in the sidebar or add `data/telco_customer_churn.csv`. Charts below use the supplied dataset only; no sample statistics are shown.")
    else:
        left, right = st.columns(2, gap="large")
        with left:
            st.subheader("Churn distribution")
            counts = data["Churn"].str.title().value_counts().reindex(["No", "Yes"], fill_value=0).rename_axis("Churn").reset_index(name="Customers")
            fig = px.bar(counts, x="Churn", y="Customers", color="Churn", color_discrete_map={"No": "#36d6c2", "Yes": "#ff8a72"}, text="Customers")
            fig.update_layout(showlegend=False)
            st.plotly_chart(chart_layout(fig), use_container_width=True)

        with right:
            st.subheader("Churn rate by contract")
            rates = data.assign(_churn=data["Churn"].str.lower().eq("yes")).groupby("Contract", as_index=False)["_churn"].mean()
            rates["Churn rate"] = rates["_churn"] * 100
            fig = px.bar(rates, x="Contract", y="Churn rate", color="Contract", color_discrete_sequence=["#36d6c2", "#60a9bd", "#ff8a72"], text_auto=".1f")
            fig.update_layout(showlegend=False, yaxis_title="Customers who churned (%)")
            st.plotly_chart(chart_layout(fig), use_container_width=True)

        st.subheader("Tenure and churn")
        tenure_data = data.dropna(subset=["tenure"]).copy()
        tenure_data["Tenure band"] = pd.cut(tenure_data["tenure"], bins=[-1, 12, 24, 48, 72, float("inf")], labels=["0–12", "13–24", "25–48", "49–72", "73+"])
        tenure_rates = tenure_data.assign(_churn=tenure_data["Churn"].str.lower().eq("yes")).groupby("Tenure band", observed=False, as_index=False)["_churn"].mean()
        tenure_rates["Churn rate"] = tenure_rates["_churn"] * 100
        fig = px.line(tenure_rates, x="Tenure band", y="Churn rate", markers=True)
        fig.update_traces(line_color="#36d6c2", marker=dict(size=9, color="#36d6c2"))
        fig.update_layout(yaxis_title="Customers who churned (%)")
        st.plotly_chart(chart_layout(fig, 300), use_container_width=True)

    st.subheader("Model feature importance")
    importance = feature_importances(bundle) if bundle is not None else None
    if importance is None:
        st.info("Feature importance will appear when a compatible fitted Decision Tree bundle is loaded. A CSV alone does not establish feature importance.")
    else:
        importance = importance.sort_values("Importance", ascending=True).tail(15)
        fig = px.bar(importance, x="Importance", y="Feature", orientation="h", color_discrete_sequence=["#36d6c2"])
        fig.update_layout(showlegend=False, yaxis_title="")
        st.plotly_chart(chart_layout(fig, max(320, len(importance) * 27)), use_container_width=True)


def render_model_lab(bundle):
    st.markdown('<div class="eyebrow">Methods and evaluation</div>', unsafe_allow_html=True)
    st.title("Model Lab")
    st.markdown(
        '<div class="panel"><h3>Decision Tree classifier</h3><p>A Decision Tree makes a sequence of feature-based splits. Each split routes a customer profile toward a leaf, where the fitted class and (when supported) class probability are returned. The model can be straightforward to explain, but its behavior depends on the training data, preprocessing, and chosen tree settings.</p><p class="subtle">This project uses the IBM Telco Customer Churn dataset. Model outputs describe patterns learned from that dataset, not a guaranteed outcome for an individual.</p></div>',
        unsafe_allow_html=True,
    )
    st.write("")
    if bundle is None:
        st.warning(f"No model bundle loaded from `{MODEL_PATH.relative_to(ROOT)}`. Evaluation values and confusion matrix are therefore unavailable; this page does not invent them.")
        metrics = {}
        evaluation = {}
    else:
        metrics = bundle.get("metrics", {}) or {}
        evaluation = bundle.get("evaluation", {}) or {}
        if not metrics:
            st.info("The model is loaded, but the bundle contains no evaluation metrics. Add scores from your actual held-out test run to display them here.")

    st.subheader("Evaluation")
    columns = st.columns(4)
    for column, (key, label) in zip(columns, [("accuracy", "Accuracy"), ("precision", "Precision"), ("recall", "Recall"), ("f1", "F1-score")]):
        column.metric(label, metric_value(metrics, key))
    test_size = evaluation.get("test_size", bundle.get("test_size") if bundle else None)
    if test_size is None:
        st.caption("Training/testing split: not recorded in the model bundle.")
    else:
        try:
            fraction = float(test_size)
            split_text = f"{(1 - fraction):.0%} training / {fraction:.0%} testing" if 0 < fraction < 1 else str(test_size)
        except (TypeError, ValueError):
            split_text = str(test_size)
        st.caption(f"Training/testing split: {split_text} (as recorded in the bundle).")

    st.subheader("Confusion matrix")
    matrix = bundle.get("confusion_matrix") if bundle else None
    if matrix is None:
        st.info("No confusion matrix was included with the saved model. Export the matrix calculated from held-out test labels and predictions to show it here.")
    else:
        try:
            values = pd.DataFrame(matrix, index=["Actual: No", "Actual: Yes"], columns=["Predicted: No", "Predicted: Yes"])
            fig = go.Figure(data=go.Heatmap(z=values.values, x=values.columns, y=values.index, colorscale=[[0, "#102632"], [1, "#36d6c2"]], text=values.values, texttemplate="%{text}", showscale=False))
            st.plotly_chart(chart_layout(fig, 280), use_container_width=True)
        except Exception as error:
            st.error(f"The confusion matrix in the bundle could not be displayed: {error}")


@st.cache_resource
def cached_load_bundle(path, modified_time):
    return load_bundle(path, modified_time)


def main():
    st.sidebar.markdown('<div class="brand">Churn<span>Sense</span> AI</div>', unsafe_allow_html=True)
    st.sidebar.caption("Customer churn intelligence")
    page = st.sidebar.radio("Navigate", PAGES, label_visibility="collapsed")
    st.sidebar.markdown("---")
    st.sidebar.subheader("Project data")
    uploaded_file = st.sidebar.file_uploader("Upload IBM Telco CSV", type=["csv"], help="Expected columns include Churn, Contract, tenure, and MonthlyCharges.")
    st.sidebar.caption(f"Local dataset path: `{DATA_PATH.relative_to(ROOT)}`")

    data = None
    source = None
    try:
        loaded, source = load_dataset(uploaded_file)
        if loaded is not None:
            data = clean_dataset(loaded)
            st.sidebar.success(f"Dataset ready · {len(data):,} rows")
    except Exception as error:
        st.sidebar.error(f"Dataset could not be loaded: {error}")

    bundle = None
    if MODEL_PATH.exists():
        try:
            bundle = cached_load_bundle(str(MODEL_PATH), MODEL_PATH.stat().st_mtime)
            st.sidebar.success("Trained model bundle loaded")
        except Exception as error:
            st.sidebar.error(f"Model bundle could not be loaded: {error}")
    else:
        st.sidebar.warning("Model bundle not installed")
        st.sidebar.caption("Expected: `models/churn_model_bundle.joblib`")

    if source and data is not None:
        st.sidebar.caption(f"Using data from {source}.")

    if page == "Overview":
        render_overview(data)
    elif page == "Predict Customer":
        render_prediction(bundle)
    elif page == "Customer Insights":
        render_insights(data, bundle)
    else:
        render_model_lab(bundle)

    st.sidebar.markdown("---")
    st.sidebar.caption("ChurnSense AI · College mini-project")


if __name__ == "__main__":
    main()