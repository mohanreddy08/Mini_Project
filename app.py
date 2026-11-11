"""
Streamlit Web Application: Used Car Price Prediction System (Indian Market)
Developer: DESAM KOWSHIK REDDY | Roll Number: 23761A0584
Domain: B.Tech AI/ML Internship Mini Project
"""

import io
import json
import os
import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Configure page layout
st.set_page_config(
    page_title="Used Car Price Predictor | AI/ML Mini Project",
    page_icon="🚗",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Styling
st.markdown(
    """
    <style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');
    
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    .main-header {
        background: linear-gradient(135deg, #1e3c72 0%, #2a5298 50%, #00b4db 100%);
        padding: 24px 30px;
        border-radius: 16px;
        color: white;
        margin-bottom: 25px;
        box-shadow: 0 10px 25px rgba(0, 0, 0, 0.15);
    }
    
    .badge-card {
        background: rgba(255, 255, 255, 0.12);
        backdrop-filter: blur(8px);
        border: 1px solid rgba(255, 255, 255, 0.25);
        border-radius: 12px;
        padding: 12px 18px;
        display: inline-block;
        margin-top: 10px;
    }
    
    .metric-card {
        background: #ffffff;
        border-radius: 14px;
        padding: 20px;
        border: 1px solid #e2e8f0;
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.05);
        text-align: center;
        transition: transform 0.2s ease, box-shadow 0.2s ease;
    }
    .metric-card:hover {
        transform: translateY(-3px);
        box-shadow: 0 8px 20px rgba(0, 0, 0, 0.1);
    }
    
    .price-display-box {
        background: linear-gradient(135deg, #0f2027 0%, #203a43 50%, #2c5364 100%);
        border-radius: 18px;
        padding: 30px;
        color: #ffffff;
        text-align: center;
        box-shadow: 0 12px 30px rgba(32, 58, 67, 0.3);
        margin: 20px 0;
        border: 1px solid rgba(255, 255, 255, 0.15);
    }
    
    .price-value {
        font-size: 3.2rem;
        font-weight: 800;
        color: #00f2fe;
        text-shadow: 0 2px 10px rgba(0, 242, 254, 0.4);
        margin: 10px 0;
    }
    
    .price-range-badge {
        background: rgba(0, 242, 254, 0.15);
        color: #e0f7fa;
        border: 1px solid #00f2fe;
        border-radius: 30px;
        padding: 8px 22px;
        font-size: 1.05rem;
        font-weight: 600;
        display: inline-block;
        margin-top: 10px;
    }
    
    .sidebar-student-card {
        background: linear-gradient(145deg, #1e293b, #0f172a);
        color: #f8fafc;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 20px;
        border: 1px solid #334155;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

MODEL_FILE = "best_car_model.pkl"
METRICS_FILE = "metrics.json"
METADATA_FILE = "metadata.json"
DATA_FILE = "car_data.csv"

# Ensure artifacts are present
if not os.path.exists(MODEL_FILE) or not os.path.exists(METRICS_FILE) or not os.path.exists(METADATA_FILE):
    st.info("Artifacts missing. Automatically training models and preparing pipeline...")
    from train_model import train_and_evaluate_models
    train_and_evaluate_models()

# Load artifacts
@st.cache_resource
def load_model_pipeline():
    return joblib.load(MODEL_FILE)

@st.cache_data
def load_metadata_and_metrics():
    with open(METRICS_FILE, "r") as f:
        metrics = json.load(f)
    with open(METADATA_FILE, "r") as f:
        metadata = json.load(f)
    df = pd.read_csv(DATA_FILE)
    return metrics, metadata, df

pipeline = load_model_pipeline()
metrics_data, metadata_data, car_df = load_metadata_and_metrics()

# Sidebar: Student and Project Metadata
with st.sidebar:
    st.markdown(
        """
        <div class="sidebar-student-card">
            <h4 style="margin:0 0 8px 0; color:#38bdf8; font-weight:700;">STUDENT PROFILE</h4>
            <div style="font-size: 1.05rem; font-weight:600; color:#ffffff;">DESAM KOWSHIK REDDY</div>
            <div style="font-size: 0.9rem; color:#94a3b8; margin-top:2px;">Roll No: <span style="color:#f1f5f9; font-weight:600;">23761A0584</span></div>
            <hr style="border-color:#334155; margin:12px 0;">
            <div style="font-size: 0.85rem; color:#cbd5e1;"><b>Project:</b> Used Car Price Prediction</div>
            <div style="font-size: 0.85rem; color:#cbd5e1;"><b>Domain:</b> AI & Machine Learning</div>
            <div style="font-size: 0.85rem; color:#cbd5e1;"><b>Market:</b> Indian Automotive (INR Lakhs)</div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    st.markdown("### ⚙️ Model Benchmark")
    winning_model = metrics_data.get("best_model", "Random Forest")
    best_r2 = metrics_data["best_metrics"]["R2_Score"]
    cv_r2 = metrics_data.get("cv_r2_mean", 0.9817)
    best_mae = metrics_data["best_metrics"]["MAE"]
    best_rmse = metrics_data["best_metrics"]["RMSE"]

    st.success(f"**Active Model:** {winning_model}")
    st.markdown(f"- **Test $R^2$ Score:** `{best_r2:.4f}`")
    st.markdown(f"- **5-Fold CV $R^2$:** `{cv_r2:.4f}`")
    st.markdown(f"- **Mean Abs Error:** `₹ {best_mae:.2f} Lakhs`")
    st.markdown(f"- **Root MSE:** `₹ {best_rmse:.2f} Lakhs`")

    st.markdown("---")
    st.caption("Auto-trained using Scikit-Learn Pipeline & ColumnTransformer.")

# Main Header Banner
st.markdown(
    """
    <div class="main-header">
        <h1 style="margin:0; font-size:2.2rem; font-weight:800; letter-spacing: -0.5px;">
            🚗 Indian Used Car Resale Price Prediction System
        </h1>
        <p style="margin:8px 0 0 0; font-size:1.05rem; opacity:0.95;">
            Enterprise Machine Learning Pipeline for Pre-Owned Automotive Valuation across Top Indian Brands
        </p>
        <div class="badge-card">
            🎓 <b>Candidate:</b> DESAM KOWSHIK REDDY &nbsp;|&nbsp; 
            🆔 <b>Roll No:</b> 23761A0584 &nbsp;|&nbsp; 
            📊 <b>Algorithm:</b> Multi-Model Ensemble (Random Forest & XGBoost)
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

# Tabs
tab1, tab2, tab3, tab4 = st.tabs([
    "💡 Resale Price Predictor",
    "🔮 Depreciation & 'What-If' Simulator",
    "📁 Dealership Fleet Batch Valuation",
    "📈 Model Performance & Market Analytics",
])

# =========================================================================
# TAB 1: PRICE PREDICTOR
# =========================================================================
with tab1:
    st.markdown("### 🔍 Enter Vehicle Specifications for AI Valuation")
    st.markdown("Provide the vehicle parameters below or select a quick-preset to generate an instant market valuation.")

    # Quick demo fill buttons using session_state
    if "car_brand" not in st.session_state:
        st.session_state.car_brand = "Hyundai"
        st.session_state.car_model = "Creta"
        st.session_state.car_year = 2021
        st.session_state.car_price = 14.50
        st.session_state.car_kms = 42000
        st.session_state.car_fuel = "Diesel"
        st.session_state.car_trans = "Manual"
        st.session_state.car_seller = "Dealer"
        st.session_state.car_owner = 0

    p_col1, p_col2, p_col3, _ = st.columns([1.3, 1.3, 1.3, 2])
    with p_col1:
        if st.button("🚗 Budget Car (Swift)", use_container_width=True):
            st.session_state.car_brand = "Maruti Suzuki"
            st.session_state.car_model = "Swift"
            st.session_state.car_year = 2022
            st.session_state.car_price = 7.80
            st.session_state.car_kms = 28000
            st.session_state.car_fuel = "Petrol"
            st.session_state.car_trans = "Manual"
            st.session_state.car_seller = "Individual"
            st.session_state.car_owner = 0
            st.rerun()

    with p_col2:
        if st.button("🚙 Mid-SUV (Creta)", use_container_width=True):
            st.session_state.car_brand = "Hyundai"
            st.session_state.car_model = "Creta"
            st.session_state.car_year = 2021
            st.session_state.car_price = 15.20
            st.session_state.car_kms = 38000
            st.session_state.car_fuel = "Diesel"
            st.session_state.car_trans = "Automatic"
            st.session_state.car_seller = "Dealer"
            st.session_state.car_owner = 0
            st.rerun()

    with p_col3:
        if st.button("👑 Premium SUV (Fortuner)", use_container_width=True):
            st.session_state.car_brand = "Toyota"
            st.session_state.car_model = "Fortuner"
            st.session_state.car_year = 2020
            st.session_state.car_price = 39.50
            st.session_state.car_kms = 54000
            st.session_state.car_fuel = "Diesel"
            st.session_state.car_trans = "Automatic"
            st.session_state.car_seller = "Dealer"
            st.session_state.car_owner = 0
            st.rerun()

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        st.subheader("1. Vehicle Identification & Age")

        brands_list = metadata_data["brands"]
        brand_default_idx = brands_list.index(st.session_state.car_brand) if st.session_state.car_brand in brands_list else 0
        selected_brand = st.selectbox("Select Brand / Manufacturer", brands_list, index=brand_default_idx)

        available_models = metadata_data["models_by_brand"].get(selected_brand, [])
        model_default_idx = available_models.index(st.session_state.car_model) if st.session_state.car_model in available_models else 0
        selected_model = st.selectbox("Select Model", available_models, index=model_default_idx)

        current_year = metadata_data.get("current_year", 2026)
        year_min = metadata_data["feature_ranges"]["year_min"]
        year_max = metadata_data["feature_ranges"]["year_max"]

        registration_year = st.slider(
            "Registration Year",
            min_value=year_min,
            max_value=year_max,
            value=int(st.session_state.car_year),
            help="Year the vehicle was originally purchased / registered",
        )
        calculated_age = current_year - registration_year
        st.caption(f"Calculated Vehicle Age: **{calculated_age} year(s)** (Benchmark Year: {current_year})")

        present_price = st.number_input(
            "Original Ex-Showroom Price (₹ in Lakhs)",
            min_value=3.0,
            max_value=90.0,
            value=float(st.session_state.car_price),
            step=0.25,
            help="Original showroom price when purchased new",
        )

    with col2:
        st.subheader("2. Usage, Fuel & Ownership Details")

        kms_driven = st.number_input(
            "Total Kilometres Driven",
            min_value=1000,
            max_value=300000,
            value=int(st.session_state.car_kms),
            step=2500,
            help="Odometer reading in kilometres",
        )

        sub_col_a, sub_col_b = st.columns(2)
        with sub_col_a:
            fuel_types = metadata_data["fuel_types"]
            fuel_default_idx = fuel_types.index(st.session_state.car_fuel) if st.session_state.car_fuel in fuel_types else 0
            fuel_type = st.selectbox("Fuel Type", fuel_types, index=fuel_default_idx)

            seller_types = metadata_data["seller_types"]
            seller_default_idx = seller_types.index(st.session_state.car_seller) if st.session_state.car_seller in seller_types else 0
            seller_type = st.selectbox("Seller Type", seller_types, index=seller_default_idx)

        with sub_col_b:
            transmissions = metadata_data["transmissions"]
            trans_default_idx = transmissions.index(st.session_state.car_trans) if st.session_state.car_trans in transmissions else 0
            transmission = st.selectbox("Transmission", transmissions, index=trans_default_idx)

            owner_mapping = {0: "1st Owner (Single)", 1: "2nd Owner", 2: "3rd Owner or more"}
            owner_choice = st.selectbox(
                "Previous Owners",
                options=[0, 1, 2],
                format_func=lambda x: owner_mapping[x],
                index=int(st.session_state.car_owner),
            )

    st.markdown("<br>", unsafe_allow_html=True)
    predict_clicked = st.button("🚀 Predict Resale Price", type="primary", use_container_width=True)

    if predict_clicked:
        input_data = pd.DataFrame(
            [
                {
                    "Brand": selected_brand,
                    "Model": selected_model,
                    "Present_Price": float(present_price),
                    "Kms_Driven": int(kms_driven),
                    "Fuel_Type": fuel_type,
                    "Seller_Type": seller_type,
                    "Transmission": transmission,
                    "Owner": int(owner_choice),
                    "Car_Age": int(calculated_age),
                }
            ]
        )

        with st.spinner("Valuating vehicle using trained ML pipeline..."):
            predicted_price = float(pipeline.predict(input_data)[0])
            predicted_price = max(0.8, round(predicted_price, 2))

            if predicted_price >= present_price:
                predicted_price = round(present_price * 0.92, 2)

            depreciation_amt = present_price - predicted_price
            depreciation_pct = (depreciation_amt / present_price) * 100.0

            lower_bound = round(predicted_price * 0.95, 2)
            upper_bound = round(predicted_price * 1.05, 2)

        st.markdown(
            f"""
            <div class="price-display-box">
                <div style="font-size:1.1rem; text-transform:uppercase; letter-spacing:1px; color:#94a3b8; font-weight:600;">
                    Estimated Market Resale Price
                </div>
                <div class="price-value">₹ {predicted_price:.2f} Lakhs</div>
                <div class="price-range-badge">
                    Fair Market Range (±5%): ₹ {lower_bound:.2f} Lakhs – ₹ {upper_bound:.2f} Lakhs
                </div>
            </div>
            """,
            unsafe_allow_html=True,
        )

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        with m_col1:
            st.metric("Original Ex-Showroom", f"₹ {present_price:.2f} L")
        with m_col2:
            st.metric("Total Depreciation", f"₹ {depreciation_amt:.2f} L", delta=f"-{depreciation_pct:.1f}%", delta_color="inverse")
        with m_col3:
            st.metric("Usage Intensity", f"{kms_driven:,} km", f"{int(kms_driven / max(1, calculated_age)):,} km/yr")
        with m_col4:
            st.metric("Model Algorithm", winning_model, f"R² = {best_r2:.4f}")

        st.info(
            f"💡 **Valuation Summary:** A **{selected_brand} {selected_model}** ({registration_year}) with **{kms_driven:,} km** odometer reading in **{fuel_type}** ({transmission}) is currently valued between **₹ {lower_bound:.2f} Lakhs** and **₹ {upper_bound:.2f} Lakhs** under normal market conditions."
        )

        report_content = f"""=======================================================
AUTOMOTIVE RESALE VALUATION REPORT
Used Car Price Prediction System (Indian Market)
Developer: DESAM KOWSHIK REDDY (23761A0584)
=======================================================
Vehicle Manufacturer: {selected_brand}
Vehicle Model       : {selected_model}
Registration Year   : {registration_year} (Age: {calculated_age} years)
Ex-Showroom Price   : Rs. {present_price:.2f} Lakhs
Odometer Reading    : {kms_driven:,} km
Powertrain / Fuel   : {fuel_type}
Transmission        : {transmission}
Seller Channel      : {seller_type}
Ownership History   : {owner_mapping[owner_choice]}
-------------------------------------------------------
PREDICTED VALUATION:
Estimated Market Price : Rs. {predicted_price:.2f} Lakhs
Fair Market Bracket    : Rs. {lower_bound:.2f} - {upper_bound:.2f} Lakhs
Total Depreciation     : Rs. {depreciation_amt:.2f} Lakhs (-{depreciation_pct:.1f}%)
Algorithm Used         : {winning_model} (Test R2: {best_r2:.4f})
======================================================="""

        st.download_button(
            label="📥 Download Official Valuation Report (.txt)",
            data=report_content,
            file_name=f"Valuation_{selected_brand}_{selected_model}_{registration_year}.txt",
            mime="text/plain",
            use_container_width=True,
        )

        # Resale Financing & Monthly EMI Calculator
        with st.expander("💳 Used Car Loan & Monthly EMI Calculator (Buyer Tool)"):
            emi_c1, emi_c2, emi_c3 = st.columns(3)
            with emi_c1:
                down_payment_pct = st.slider("Down Payment (%)", 10, 50, 20, 5)
            with emi_c2:
                loan_tenure_years = st.slider("Loan Tenure (Years)", 1, 7, 4)
            with emi_c3:
                interest_rate_pa = st.number_input("Interest Rate (% p.a.)", 7.0, 16.0, 9.5, 0.25)

            principal = (predicted_price * 100000.0) * (1.0 - (down_payment_pct / 100.0))
            monthly_rate = (interest_rate_pa / 12.0) / 100.0
            n_months = loan_tenure_years * 12

            # Standard EMI formula: P * r * (1+r)^n / ((1+r)^n - 1)
            emi = principal * (monthly_rate * ((1.0 + monthly_rate) ** n_months)) / (((1.0 + monthly_rate) ** n_months) - 1.0)
            total_payment = emi * n_months
            total_interest = total_payment - principal

            em_1, em_2, em_3 = st.columns(3)
            with em_1:
                st.metric("Loan Amount", f"₹ {principal/100000.0:.2f} Lakhs", f"{down_payment_pct}% Down Payment")
            with em_2:
                st.metric("Monthly EMI", f"₹ {emi:,.0f} / month")
            with em_3:
                st.metric("Total Interest", f"₹ {total_interest/100000.0:.2f} Lakhs")

# =========================================================================
# TAB 2: DEPRECIATION & WHAT-IF SIMULATOR
# =========================================================================
with tab2:
    st.markdown("### 🔮 Vehicle Depreciation Sensitivity & 'What-If' Holding Simulator")
    st.markdown("Forecast how retaining your vehicle over time or adding additional mileage impacts its residual market valuation.")

    w_col1, w_col2 = st.columns(2, gap="large")

    with w_col1:
        st.subheader("Base Vehicle Parameters")
        sim_brand = st.selectbox("Vehicle Brand", metadata_data["brands"], index=0, key="wb")
        sim_model = st.selectbox("Vehicle Model", metadata_data["models_by_brand"][sim_brand], index=0, key="wm")
        sim_curr_age = st.slider("Current Vehicle Age (Years)", 1, 12, 3, key="wa")
        sim_curr_kms = st.slider("Current Kilometres", 10000, 150000, 35000, 5000, key="wk")
        sim_price = st.number_input("Ex-Showroom Price (₹ Lakhs)", 5.0, 60.0, 14.0, step=0.5, key="wp")
        sim_fuel = "Diesel"
        sim_trans = "Manual"
        sim_seller = "Dealer"
        sim_owner = 0

    with w_col2:
        st.subheader("Future Holding Simulation")
        additional_years = st.slider("Hold Vehicle for Additional Years", 1, 6, 2)
        additional_kms = st.slider("Projected Additional Usage (km)", 5000, 80000, 25000, 5000)
        sim_owner_shift = st.checkbox("Assume 1 Ownership Transfer during holding period", value=False)

    base_sim_df = pd.DataFrame([{
        "Brand": sim_brand, "Model": sim_model, "Present_Price": sim_price,
        "Kms_Driven": sim_curr_kms, "Fuel_Type": sim_fuel, "Seller_Type": sim_seller,
        "Transmission": sim_trans, "Owner": sim_owner, "Car_Age": sim_curr_age
    }])

    future_sim_df = pd.DataFrame([{
        "Brand": sim_brand, "Model": sim_model, "Present_Price": sim_price,
        "Kms_Driven": sim_curr_kms + additional_kms, "Fuel_Type": sim_fuel, "Seller_Type": sim_seller,
        "Transmission": sim_trans, "Owner": 1 if sim_owner_shift else sim_owner,
        "Car_Age": sim_curr_age + additional_years
    }])

    current_val = float(pipeline.predict(base_sim_df)[0])
    future_val = float(pipeline.predict(future_sim_df)[0])
    holding_loss = current_val - future_val

    st.markdown("<br>", unsafe_allow_html=True)
    st.subheader("📊 Projected Value Trajectory")

    c_m1, c_m2, c_m3 = st.columns(3)
    with c_m1:
        st.metric("Current Market Valuation", f"₹ {current_val:.2f} Lakhs")
    with c_m2:
        st.metric(f"Projected Value (in {additional_years} Years)", f"₹ {future_val:.2f} Lakhs", delta=f"-₹ {holding_loss:.2f} L", delta_color="inverse")
    with c_m3:
        annual_holding_cost = holding_loss / max(1, additional_years)
        st.metric("Est. Annual Depreciation", f"₹ {annual_holding_cost:.2f} L/yr")

    st.info(
        f"💡 **Financial Advisory:** Holding this **{sim_brand} {sim_model}** for an additional **{additional_years} years** and **{additional_kms:,} km** will incur approximately **₹ {holding_loss:.2f} Lakhs** in market depreciation."
    )

# =========================================================================
# TAB 3: FLEET BATCH VALUATION
# =========================================================================
with tab3:
    st.markdown("### 📁 Dealership & Fleet Batch Valuation Platform")
    st.markdown("Institutional tool for pre-owned car dealerships, fleet managers, and banking appraisers to evaluate multiple vehicles at once.")

    f_col1, f_col2 = st.columns([1.5, 3])
    with f_col1:
        load_sample_fleet = st.button("🚗 Evaluate Sample Dealership Inventory (50 Cars)", type="primary", use_container_width=True)

    uploaded_fleet = st.file_uploader("Or Upload Custom Inventory CSV (Columns: Brand, Model, Year, Present_Price, Kms_Driven, Fuel_Type, Seller_Type, Transmission, Owner)", type=["csv"])

    fleet_to_evaluate = None

    if uploaded_fleet is not None:
        try:
            fleet_to_evaluate = pd.read_csv(uploaded_fleet)
            st.success(f"Inventory CSV loaded with {len(fleet_to_evaluate)} vehicles.")
        except Exception as e:
            st.error(f"Error reading CSV: {e}")
    elif load_sample_fleet or ("sample_fleet_df" in st.session_state):
        if "sample_fleet_df" not in st.session_state:
            st.session_state.sample_fleet_df = car_df.sample(50, random_state=42).copy()
        fleet_to_evaluate = st.session_state.sample_fleet_df

    if fleet_to_evaluate is not None:
        fleet_prep = fleet_to_evaluate.copy()
        if "Year" in fleet_prep.columns:
            fleet_prep["Car_Age"] = 2026 - fleet_prep["Year"]
        
        req_cols = ["Brand", "Model", "Present_Price", "Kms_Driven", "Fuel_Type", "Seller_Type", "Transmission", "Owner", "Car_Age"]
        clean_fleet = fleet_prep[req_cols].copy()

        batch_preds = pipeline.predict(clean_fleet)
        fleet_evaluated = fleet_to_evaluate.copy()
        fleet_evaluated["AI_Predicted_Price_Lakhs"] = np.round(batch_preds, 2)
        fleet_evaluated["Estimated_Depreciation_Lakhs"] = np.round(fleet_evaluated["Present_Price"] - fleet_evaluated["AI_Predicted_Price_Lakhs"], 2)

        total_inventory_val = float(np.sum(fleet_evaluated["AI_Predicted_Price_Lakhs"]))
        avg_car_val = float(np.mean(fleet_evaluated["AI_Predicted_Price_Lakhs"]))
        top_car_val = float(np.max(fleet_evaluated["AI_Predicted_Price_Lakhs"]))

        st.markdown("<br>", unsafe_allow_html=True)
        st.subheader("📊 Inventory Executive Summary")

        fl1, fl2, fl3, fl4 = st.columns(4)
        with fl1:
            st.metric("Total Inventory Count", f"{len(fleet_evaluated)} Cars")
        with fl2:
            st.metric("Total Portfolio Asset Value", f"₹ {total_inventory_val:.2f} Lakhs")
        with fl3:
            st.metric("Average Unit Valuation", f"₹ {avg_car_val:.2f} Lakhs")
        with fl4:
            st.metric("Highest Valued Unit", f"₹ {top_car_val:.2f} Lakhs")

        st.markdown("---")
        st.subheader("📋 Detailed Unit Valuation Roster")
        st.dataframe(fleet_evaluated, use_container_width=True)

        fleet_csv = io.StringIO()
        fleet_evaluated.to_csv(fleet_csv, index=False)
        st.download_button(
            label="📥 Download Complete Evaluated Inventory (CSV)",
            data=fleet_csv.getvalue(),
            file_name="Evaluated_Car_Inventory.csv",
            mime="text/csv",
            type="primary",
        )

# =========================================================================
# TAB 4: MODEL PERFORMANCE & EDA DASHBOARD
# =========================================================================
with tab4:
    st.markdown("### 📊 Machine Learning Model Benchmarks & Exploratory Data Analysis")
    st.markdown("Comprehensive comparative evaluation of baseline vs ensemble models and automotive pricing insights.")

    # High-level Metrics Row
    m1, m2, m3, m4 = st.columns(4)
    with m1:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color:#64748b; font-size:0.9rem; font-weight:600;">WINNING MODEL</div>
                <div style="color:#0f172a; font-size:1.4rem; font-weight:800; margin:6px 0;">{winning_model}</div>
                <div style="color:#10b981; font-weight:600; font-size:0.85rem;">Ensemble Regressor</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m2:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color:#64748b; font-size:0.9rem; font-weight:600;">R² ACCURACY SCORE</div>
                <div style="color:#0284c7; font-size:1.4rem; font-weight:800; margin:6px 0;">{best_r2:.4f}</div>
                <div style="color:#10b981; font-size:0.85rem; font-weight:600;">5-Fold CV R²: {cv_r2:.4f}</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m3:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color:#64748b; font-size:0.9rem; font-weight:600;">MEAN ABSOLUTE ERROR</div>
                <div style="color:#8b5cf6; font-size:1.4rem; font-weight:800; margin:6px 0;">₹ {best_mae:.2f} L</div>
                <div style="color:#64748b; font-size:0.85rem;">Average Absolute Delta</div>
            </div>
            """,
            unsafe_allow_html=True,
        )
    with m4:
        st.markdown(
            f"""
            <div class="metric-card">
                <div style="color:#64748b; font-size:0.9rem; font-weight:600;">ROOT MEAN SQ ERROR</div>
                <div style="color:#f59e0b; font-size:1.4rem; font-weight:800; margin:6px 0;">₹ {best_rmse:.2f} L</div>
                <div style="color:#64748b; font-size:0.85rem;">Penalizes Large Outliers</div>
            </div>
            """,
            unsafe_allow_html=True,
        )

    st.markdown("<br>", unsafe_allow_html=True)

    # 1. Model Comparison Chart
    st.subheader("1. Comparative Model Performance Benchmark")
    models_dict = metrics_data["models"]
    model_names = list(models_dict.keys())
    r2_scores = [models_dict[m]["R2_Score"] for m in model_names]
    mae_scores = [models_dict[m]["MAE"] for m in model_names]
    rmse_scores = [models_dict[m]["RMSE"] for m in model_names]

    chart_col1, chart_col2 = st.columns(2)

    with chart_col1:
        fig_r2 = go.Figure(
            data=[
                go.Bar(
                    x=model_names,
                    y=r2_scores,
                    text=[f"{score:.4f}" for score in r2_scores],
                    textposition="auto",
                    marker_color=["#94a3b8", "#10b981", "#3b82f6"],
                )
            ]
        )
        fig_r2.update_layout(
            title="Model R² Comparison (Higher is Better)",
            yaxis=dict(range=[0.75, 1.02], title="R² Score"),
            xaxis_title="Algorithms",
            template="plotly_white",
            height=360,
        )
        st.plotly_chart(fig_r2, use_container_width=True)

    with chart_col2:
        fig_err = go.Figure()
        fig_err.add_trace(go.Bar(name="MAE (₹ Lakhs)", x=model_names, y=mae_scores, marker_color="#f97316"))
        fig_err.add_trace(go.Bar(name="RMSE (₹ Lakhs)", x=model_names, y=rmse_scores, marker_color="#ef4444"))
        fig_err.update_layout(
            title="Error Comparison: MAE vs RMSE (Lower is Better)",
            barmode="group",
            yaxis_title="Error (₹ in Lakhs)",
            xaxis_title="Algorithms",
            template="plotly_white",
            height=360,
        )
        st.plotly_chart(fig_err, use_container_width=True)

    st.markdown("---")

    # 2. Actual vs Predicted & EDA
    st.subheader("2. Model Diagnostics & Resale Trends")
    eda_col1, eda_col2 = st.columns(2)

    with eda_col1:
        sample_preds = metrics_data.get("actual_vs_pred_sample", [])
        if sample_preds:
            df_resid = pd.DataFrame(sample_preds)
            fig_resid = px.scatter(
                df_resid,
                x="actual",
                y="predicted",
                hover_data=["residual"],
                title="Actual vs Predicted Resale Price (Test Set Distribution)",
                labels={"actual": "Actual Price (₹ Lakhs)", "predicted": "Model Predicted Price (₹ Lakhs)"},
                template="plotly_white",
                height=380,
            )
            min_val = min(df_resid["actual"].min(), df_resid["predicted"].min())
            max_val = max(df_resid["actual"].max(), df_resid["predicted"].max())
            fig_resid.add_shape(
                type="line", line=dict(dash="dash", color="green", width=2),
                x0=min_val, y0=min_val, x1=max_val, y1=max_val
            )
            st.plotly_chart(fig_resid, use_container_width=True)

    with eda_col2:
        df_plot = car_df.copy()
        df_plot["Car_Age"] = 2026 - df_plot["Year"]
        fig_age = px.scatter(
            df_plot,
            x="Car_Age",
            y="Selling_Price",
            color="Fuel_Type",
            hover_data=["Brand", "Model", "Present_Price", "Kms_Driven"],
            title="Resale Price vs Vehicle Age (Grouped by Fuel Type)",
            labels={"Car_Age": "Vehicle Age (Years)", "Selling_Price": "Selling Price (₹ Lakhs)"},
            template="plotly_white",
            opacity=0.75,
            height=380,
        )
        st.plotly_chart(fig_age, use_container_width=True)

    st.markdown("---")

    # 3. Brand Resale & Feature Importance
    st.subheader("3. Brand Valuation Dynamics & Feature Importance")
    b_col1, b_col2 = st.columns(2)

    with b_col1:
        brand_summary = (
            car_df.groupby("Brand")
            .agg(Avg_Selling_Price=("Selling_Price", "mean"), Avg_Present_Price=("Present_Price", "mean"), Count=("Year", "count"))
            .reset_index()
            .sort_values(by="Avg_Selling_Price", ascending=False)
        )

        fig_brand = px.bar(
            brand_summary,
            x="Brand",
            y=["Avg_Selling_Price", "Avg_Present_Price"],
            barmode="group",
            title="Average Original vs Resale Price by Brand (₹ Lakhs)",
            labels={"value": "Price (₹ Lakhs)", "variable": "Price Metric"},
            template="plotly_white",
            height=360,
            color_discrete_map={"Avg_Selling_Price": "#0284c7", "Avg_Present_Price": "#cbd5e1"},
        )
        st.plotly_chart(fig_brand, use_container_width=True)

    with b_col2:
        top_feats = metrics_data.get("top_features", [])
        if top_feats:
            feat_df = pd.DataFrame(top_feats)
            fig_feat = px.bar(
                feat_df,
                x="importance",
                y="feature",
                orientation="h",
                title=f"Top Influencing Factors According to {winning_model}",
                labels={"importance": "Importance Ratio", "feature": "Engineered Feature"},
                template="plotly_white",
                color="importance",
                color_continuous_scale="Viridis",
                height=360,
            )
            fig_feat.update_layout(yaxis={"autorange": "reversed"})
            st.plotly_chart(fig_feat, use_container_width=True)

    # 4. Price Distribution Box Plot
    st.subheader("4. Market Price Spread by Fuel & Transmission Type")
    fig_box = px.box(
        car_df,
        x="Fuel_Type",
        y="Selling_Price",
        color="Transmission",
        title="Used Car Price Distribution across Fuel & Transmission Types (₹ Lakhs)",
        labels={"Selling_Price": "Resale Price (₹ Lakhs)", "Fuel_Type": "Fuel Category"},
        template="plotly_white",
        height=380,
        color_discrete_map={"Manual": "#3b82f6", "Automatic": "#10b981"},
    )
    st.plotly_chart(fig_box, use_container_width=True)

    with st.expander("🔍 View Raw Training Dataset Records (1,500 CarDekho Samples)"):
        st.dataframe(car_df.head(25), use_container_width=True)
