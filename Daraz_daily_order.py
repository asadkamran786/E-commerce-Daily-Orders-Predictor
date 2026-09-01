import streamlit as st
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_absolute_error

st.set_page_config(
    page_title="E-commerce Order Predictor",
    page_icon="🛒",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ---------- Professional dashboard styling ----------
st.markdown("""
<style>
.stApp { background: #f5f7fb; }
.main-title { font-size: 38px; font-weight: 800; color: #2f3542; }
.subtitle { color: #747d8c; font-size: 16px; margin-bottom: 25px; }
.kpi-card {
    padding: 20px; border-radius: 14px; color: white;
    min-height: 125px; box-shadow: 0 5px 15px rgba(0,0,0,.08);
}
.kpi-title { font-size: 14px; font-weight: 600; opacity: .9; }
.kpi-value { font-size: 30px; font-weight: 800; margin-top: 8px; }
.kpi-note { font-size: 12px; margin-top: 5px; opacity: .85; }
.prediction-box {
    background: linear-gradient(135deg,#ff6b35,#ff8c42);
    padding: 25px; border-radius: 16px; color: white;
    text-align: center; box-shadow: 0 8px 20px rgba(255,107,53,.25);
}
.prediction-label { font-size: 16px; font-weight: 600; }
.prediction-number { font-size: 48px; font-weight: 900; margin: 5px 0; }
.section-title { font-size: 23px; font-weight: 750; color: #2f3542; margin: 20px 0 12px; }
.info-box {
    background: white; padding: 18px; border-radius: 14px;
    border-left: 5px solid #ff6b35; box-shadow: 0 3px 12px rgba(0,0,0,.05);
}
.footer { text-align: center; color: #a4b0be; padding: 25px; font-size: 13px; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    return pd.read_csv("daraz_daily_orders_synthetic.csv")

df = load_data()

FEATURES = [
    "Website_Visitors",
    "Ad_Spend_PKR",
    "Discount_Percent",
    "Weekend"
]

X = df[FEATURES]
y = df["Daily_Orders"]

# 80/20 evaluation split
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.20, random_state=42
)

evaluation_model = LinearRegression()
evaluation_model.fit(X_train, y_train)

test_predictions = evaluation_model.predict(X_test)
mae = mean_absolute_error(y_test, test_predictions)

# Train final dashboard model on all historical records
model = LinearRegression()
model.fit(X, y)

# ---------- Header ----------
st.markdown(
    '<div class="main-title">🛒 E-commerce Daily Orders Predictor</div>',
    unsafe_allow_html=True
)
st.markdown(
    '<div class="subtitle">AI-powered dashboard for estimating daily e-commerce orders using Linear Regression</div>',
    unsafe_allow_html=True
)

# ---------- Sidebar ----------
with st.sidebar:
    st.header("🎛️ Prediction Inputs")
    st.caption("Adjust the business inputs and generate a prediction.")

    website_visitors = st.slider(
        "👥 Website Visitors",
        int(df["Website_Visitors"].min()),
        int(df["Website_Visitors"].max()),
        5000, 100
    )

    ad_spend = st.slider(
        "💰 Ad Spend (PKR)",
        int(df["Ad_Spend_PKR"].min()),
        int(df["Ad_Spend_PKR"].max()),
        20000, 500
    )

    discount = st.slider(
        "🏷️ Discount (%)", 0, 30, 20, 1
    )

    weekend_label = st.radio(
        "📅 Day Type", ["Weekday", "Weekend"], horizontal=True
    )
    weekend = 1 if weekend_label == "Weekend" else 0

    st.button("🔮 Predict Daily Orders", type="primary", use_container_width=True)

new_day = pd.DataFrame({
    "Website_Visitors": [website_visitors],
    "Ad_Spend_PKR": [ad_spend],
    "Discount_Percent": [discount],
    "Weekend": [weekend]
})

prediction = max(0, float(model.predict(new_day)[0]))

# ---------- KPI cards ----------
st.markdown('<div class="section-title">📊 Dashboard Overview</div>', unsafe_allow_html=True)

k1, k2, k3, k4 = st.columns(4)

with k1:
    st.markdown(f"""
    <div class="kpi-card" style="background:linear-gradient(135deg,#ff6b35,#ff8c42)">
    <div class="kpi-title">PREDICTED ORDERS</div>
    <div class="kpi-value">{prediction:,.0f}</div>
    <div class="kpi-note">Estimated for selected day</div>
    </div>
    """, unsafe_allow_html=True)

with k2:
    st.markdown(f"""
    <div class="kpi-card" style="background:linear-gradient(135deg,#3742fa,#5352ed)">
    <div class="kpi-title">WEBSITE VISITORS</div>
    <div class="kpi-value">{website_visitors:,.0f}</div>
    <div class="kpi-note">Selected traffic level</div>
    </div>
    """, unsafe_allow_html=True)

with k3:
    st.markdown(f"""
    <div class="kpi-card" style="background:linear-gradient(135deg,#20bf6b,#26de81)">
    <div class="kpi-title">AD SPEND</div>
    <div class="kpi-value">PKR {ad_spend:,.0f}</div>
    <div class="kpi-note">Selected advertising budget</div>
    </div>
    """, unsafe_allow_html=True)

with k4:
    st.markdown(f"""
    <div class="kpi-card" style="background:linear-gradient(135deg,#8854d0,#a55eea)">
    <div class="kpi-title">MODEL MAE</div>
    <div class="kpi-value">{mae:.1f}</div>
    <div class="kpi-note">Average test-set error</div>
    </div>
    """, unsafe_allow_html=True)

# ---------- Prediction result ----------
left, right = st.columns([1.15, 1])

with left:
    st.markdown('<div class="section-title">🎯 Prediction Result</div>', unsafe_allow_html=True)
    st.markdown(f"""
    <div class="prediction-box">
        <div class="prediction-label">Estimated Daily Orders</div>
        <div class="prediction-number">{prediction:,.0f}</div>
        <div>{weekend_label} • {discount}% discount • PKR {ad_spend:,.0f} ad spend</div>
    </div>
    """, unsafe_allow_html=True)

with right:
    st.markdown('<div class="section-title">🧾 Selected Inputs</div>', unsafe_allow_html=True)
    summary = pd.DataFrame({
        "Business Input": ["Website Visitors", "Ad Spend", "Discount", "Day Type"],
        "Selected Value": [
            f"{website_visitors:,}", f"PKR {ad_spend:,}",
            f"{discount}%", weekend_label
        ]
    })
    st.dataframe(summary, use_container_width=True, hide_index=True)

st.markdown(
    f'<div class="info-box"><b>💡 Interpretation</b><br>'
    f'The model estimates approximately <b>{prediction:,.0f} orders</b> for this day. '
    f'This is an estimate based on historical patterns, not a guaranteed future result.</div>',
    unsafe_allow_html=True
)

# ---------- Charts ----------
st.markdown('<div class="section-title">📈 Historical Analytics</div>', unsafe_allow_html=True)

tab1, tab2, tab3 = st.tabs([
    "📦 Daily Orders Trend", "👥 Visitors vs Orders", "💰 Ad Spend vs Orders"
])

with tab1:
    st.line_chart(
        df[["Day_Number", "Daily_Orders"]].set_index("Day_Number"),
        height=350
    )

with tab2:
    visitors_chart = df[["Website_Visitors", "Daily_Orders"]].sort_values("Website_Visitors")
    st.scatter_chart(
        visitors_chart, x="Website_Visitors", y="Daily_Orders", height=350
    )

with tab3:
    spend_chart = df[["Ad_Spend_PKR", "Daily_Orders"]].sort_values("Ad_Spend_PKR")
    st.scatter_chart(
        spend_chart, x="Ad_Spend_PKR", y="Daily_Orders", height=350
    )

# ---------- Model information ----------
st.markdown('<div class="section-title">🧠 Model Information</div>', unsafe_allow_html=True)

a, b, c = st.columns(3)
with a:
    st.metric("Training Records", len(X_train))
with b:
    st.metric("Testing Records", len(X_test))
with c:
    st.metric("Test MAE", f"{mae:.2f} orders")

with st.expander("🔍 View Linear Regression Coefficients"):
    coefficients = pd.DataFrame({
        "Feature": FEATURES,
        "Coefficient": evaluation_model.coef_
    })
    st.dataframe(coefficients, use_container_width=True, hide_index=True)

with st.expander("🎓 How This Model Works"):
    st.markdown("""
    **Step 1:** The model learns patterns from historical e-commerce data.

    **Step 2:** Website visitors, advertising spend, discount percentage and weekend status
    are used as input features.

    **Step 3:** Linear Regression estimates the expected Daily Orders.

    **Step 4:** MAE measures the average prediction error on unseen test data.
    """)

st.markdown(
    '<div class="footer">E-commerce Regression Classroom Project • Synthetic Dataset • Linear Regression</div>',
    unsafe_allow_html=True
)
