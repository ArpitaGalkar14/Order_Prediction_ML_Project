import streamlit as st
import pandas as pd
import joblib

# Load LightGBM model
model = joblib.load("lgb_model.pkl")

# Page configuration
st.set_page_config(
    page_title="📊 Demand Prediction Dashboard",
    page_icon="📊",
    layout="wide"
)

# ---- Custom CSS ----
st.markdown("""
<style>
[data-testid="stAppViewContainer"] {
    background: linear-gradient(135deg, #a1c4fd, #c2e9fb);
    font-family: 'Segoe UI', sans-serif;
    display: flex;
    flex-direction: column;
    align-items: center;
}
h1, h2, h3 { text-align: center; color: #1f3b73; text-shadow: 1px 1px 2px #ffffffaa; }
.stNumberInput label { display: block; text-align: center; font-weight: bold; margin-bottom: 5px; }
.stNumberInput>div { width: 50% !important; min-width: 200px; margin: 0 auto; }
.stNumberInput>div>div>input { border: 2px solid #4CAF50; border-radius: 10px; padding: 10px; font-size: 16px; transition: 0.3s; box-shadow: inset 0 1px 3px rgba(0,0,0,0.1);}
.stNumberInput>div>div>input:hover { border: 2px solid #ff7f0e; background-color: #fff9e6; box-shadow: 0 0 8px #ff7f0e33; }
.stButton>button { background: linear-gradient(90deg, #4CAF50, #45a049); color: white; font-weight: bold; padding: 0.7em 1em; font-size: 16px; border-radius: 12px; transition: 0.3s; width: 200px; box-shadow: 0 4px 8px rgba(0,0,0,0.2); }
.stButton>button:hover { background: linear-gradient(90deg, #ff7f0e, #ff9933); transform: scale(1.05); box-shadow: 0 6px 12px rgba(0,0,0,0.3);}
.stButton.reset-btn>button { background: linear-gradient(90deg, #d9534f, #c9302c);}
.stButton.reset-btn>button:hover { background: linear-gradient(90deg, #ff5c33, #ff704d); transform: scale(1.05);}
.message-card { background: linear-gradient(135deg, #ffffffcc, #e6f2ffcc); border-radius: 20px; padding: 20px; box-shadow: 0px 8px 20px rgba(0,0,0,0.25); text-align: center; font-size: 22px; color: #1f3b73; margin-bottom: 20px; transition: 0.4s; animation: fadeIn 0.6s ease-in-out; width: 50%; margin-left: auto; margin-right: auto; }
@keyframes fadeIn { 0% {opacity: 0; transform: translateY(-10px);} 100% {opacity: 1; transform: translateY(0);} }
</style>
""", unsafe_allow_html=True)

# ---- Title ----
st.title("📊 Demand Prediction Dashboard")
st.subheader("Enter feature values to predict demand:")

# ---- Feature info ----
feature_info = {
    "Store ID": ("🏬 Unique identifier for the store", 1, 10000),
    "Product ID": ("📦 Unique identifier for the product", 1, 500),
    "Day": ("📅 Day of the month (1-31)", 1, 31),
    "Month": ("🗓 Month number (1-12)", 1, 12),
    "Promotion": ("🎁 On promotion (1:Yes,0:No)", 0, 1),
    "Holiday": ("🏖 Holiday (1:Yes,0:No)", 0, 1),
    "Price": ("💲 Price of the product", 0, 10000),
    "Competitor Price": ("💰 Price of competing product", 0, 10000),
    "Stock": ("📊 Available stock in the store", 0, 10000),
    "Sales Last Week": ("📈 Units sold last week", 0, 10000),
    "Sales Last Month": ("📉 Units sold last month", 0, 10000),
    "Customer Rating": ("⭐ Average customer rating (1-5)", 1, 5),
    "Discount": ("🔖 Discount on product (%)", 0, 100),
    "Season": ("🌦 Season encoded as number", 1, 4),
    "Weather": ("⛅ Weather index (0-5)", 0, 5),
    "Events": ("🎉 Number of local events", 0, 10),
    "Ad Spend": ("📢 Advertising spend", 0, 10000),
    "Foot Traffic": ("🚶 Store foot traffic", 0, 10000),
    "Store Size": ("📐 Store size in sq.ft", 100, 10000),
    "Region ID": ("🗺 Region identifier", 1, 50)
}

# ---- Session state initialization ----
if "input_values" not in st.session_state:
    st.session_state.input_values = {key: feature_info[key][1] for key in feature_info.keys()}
if "message" not in st.session_state:
    st.session_state.message = ""

# ---- Input fields ----
for feature, (desc, min_val, max_val) in feature_info.items():
    st.session_state.input_values[feature] = st.number_input(
        f"{desc}",
        min_value=min_val,
        max_value=max_val,
        value=st.session_state.input_values[feature]
    )

# ---- Buttons horizontally centered ----
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    btn1_col, btn2_col = st.columns([1,1])
    with btn1_col:
        predict_btn = st.button("Predict")
    with btn2_col:
        reset_btn = st.button("Reset", key="reset_btn")

# ---- Prediction ----
if predict_btn:
    input_df = pd.DataFrame([st.session_state.input_values])
    prediction = model.predict(input_df)[0]
    st.session_state.message = f"🔹 Predicted Demand: {prediction:.2f} %"

# ---- Reset functionality ----
if reset_btn:
    # Reset all input values to defaults
    for key in st.session_state.input_values.keys():
        st.session_state.input_values[key] = feature_info[key][1]
    st.session_state.message = ""
    # Force rerun to reflect reset immediately
    st.experimental_rerun = None  # workaround for latest Streamlit versions
    st.stop()

# ---- Display message ----
st.markdown(f"<div class='message-card'>{st.session_state.message}</div>", unsafe_allow_html=True)
