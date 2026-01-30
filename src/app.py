import streamlit as st
import pickle
import numpy as np

# ------------------ PAGE CONFIG ------------------
st.set_page_config(
    page_title="Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

# ------------------ LOAD MODEL ------------------
model = pickle.load(open("xgb_churn_model.pkl", "rb"))

# ------------------ CUSTOM CSS ------------------
st.markdown("""
<style>

body {
    background-color: #0e1117;
}

.main {
    background: linear-gradient(145deg,#0e1117,#111827);
}

h1, h2, h3 {
    font-weight: 700 !important;
    letter-spacing: 0.5px;
}

.card {
    padding: 25px;
    border-radius: 15px;
    background: #111827;
    box-shadow: 0px 4px 20px rgba(0,0,0,0.6);
    animation: fadeIn 1.2s ease-in;
}

@keyframes fadeIn {
    from {opacity: 0; transform: translateY(20px);}
    to {opacity: 1; transform: translateY(0);}
}

.stButton>button {
    background: linear-gradient(90deg,#2563eb,#4f46e5);
    color: white;
    border-radius: 10px;
    height: 3em;
    width: 100%;
    font-size: 18px;
    font-weight: bold;
    border: none;
    transition: 0.3s;
}

.stButton>button:hover {
    transform: scale(1.05);
    background: linear-gradient(90deg,#1d4ed8,#4338ca);
}

[data-testid="metric-container"] {
    background-color: #111827;
    border-radius: 12px;
    padding: 15px;
    box-shadow: 0px 4px 12px rgba(0,0,0,0.5);
}

section[data-testid="stSidebar"] {
    background: #020617;
}

</style>
""", unsafe_allow_html=True)

# ------------------ HEADER ------------------
st.title("📊 Churn Intelligence Dashboard")
st.caption("AI-powered customer churn prediction system")

st.divider()

# ------------------ SIDEBAR ------------------
st.sidebar.header("Customer Inputs")

plan_type = st.sidebar.selectbox("Plan Type", ["Basic", "Premium"])
monthly_fee = st.sidebar.number_input("Monthly Fee", 0, 5000, 699)

avg_weekly_usage_hours = st.sidebar.slider(
    "Weekly Usage (hrs)", 0.0, 50.0, 5.0
)

support_tickets = st.sidebar.slider(
    "Support Tickets", 0, 20, 1
)

payment_failures = st.sidebar.slider(
    "Payment Failures", 0, 10, 0
)

tenure_months = st.sidebar.slider(
    "Tenure (Months)", 0, 60, 12
)

last_login_days_ago = st.sidebar.slider(
    "Last Login Gap (Days)", 0, 60, 5
)

plan_type_encoded = 1 if plan_type == "Premium" else 0

# ------------------ SUMMARY ------------------
col1, col2 = st.columns(2)

with col1:
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.subheader("Customer Profile")

    st.write(f"""
    **Plan:** {plan_type}  
    **Fee:** ₹{monthly_fee}  
    **Usage:** {avg_weekly_usage_hours} hrs/week  
    **Tickets:** {support_tickets}  
    **Failures:** {payment_failures}  
    **Tenure:** {tenure_months} months  
    """)
    st.markdown('</div>', unsafe_allow_html=True)

# ------------------ PREDICTION ------------------
if st.button("Predict Churn Risk"):

    input_data = np.array([[
        plan_type_encoded,
        monthly_fee,
        avg_weekly_usage_hours,
        support_tickets,
        payment_failures,
        tenure_months,
        last_login_days_ago
    ]])

    prediction = model.predict(input_data)[0]
    prob = model.predict_proba(input_data)[0][1]

    with col2:
        st.markdown('<div class="card">', unsafe_allow_html=True)

        st.subheader("Prediction Outcome")

        st.metric("Churn Probability", f"{prob*100:.2f}%")
        st.progress(int(prob * 100))

        if prediction == 1:
            st.error("High Churn Risk Detected")

            if prob > 0.75:
                st.warning("Critical: Immediate retention action required.")
            else:
                st.warning("Moderate risk: Monitor customer.")

        else:
            st.success("Customer Likely to Stay")

        st.markdown('</div>', unsafe_allow_html=True)

# ------------------ FOOTER ------------------
st.divider()
st.caption("© 2026 Churn Intelligence • Built with Streamlit & XGBoost")
