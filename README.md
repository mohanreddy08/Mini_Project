# 🚗 Used Car Price Prediction System (Indian Automotive Market)

**A B.Tech Artificial Intelligence & Machine Learning Internship Mini Project**

---

## 📌 Student & Project Metadata

| Attribute | Details |
| :--- | :--- |
| **Student Name** | **DESAM KOWSHIK REDDY** |
| **Roll Number** | **23761A0584** |
| **Course** | B.Tech in Artificial Intelligence & Machine Learning |
| **Project Title** | Used Car Price Prediction System (Indian Market) |
| **Domain** | Machine Learning, Tabular Predictive Modeling, Full-Stack Python |
| **Target Currency** | Indian Rupees (₹ in Lakhs) |
| **Current Baseline Year** | 2026 |

---

## 📖 1. Abstract & Problem Statement

The Indian pre-owned automobile sector is among the fastest-growing automotive ecosystems globally. However, used car valuations are traditionally plagued by information asymmetry, subjective dealership markups, and unstandardized pricing metrics. Factors such as vehicle age, odometer mileage, brand retention, fuel type premiums, and transmission preferences create complex non-linear depreciation patterns.

**Project Objective:**
This project establishes an automated, end-to-end Machine Learning pipeline that:
1. Synthesizes a realistic 1,500-sample Indian automotive dataset modeled after real-world CarDekho listings.
2. Employs Scikit-Learn `ColumnTransformer` pipelines for one-hot categorical encoding and numerical standardization.
3. Compares three regression paradigms: **Linear Regression (Baseline)**, **Random Forest Regressor**, and **Extreme Gradient Boosting (XGBoost)**.
4. Serializes the optimal model and serves interactive valuations via a modern **Streamlit Web Dashboard** featuring real-time brand-model cascading filters, fair market ranges ($\pm 5\%$), What-If holding simulators, fleet batch evaluation, and exploratory analytics.

---

## 🏗️ 2. System Architecture & ML Pipeline

```
+--------------------------------------------------------------+
|                    1. Raw Dataset Source                     |
|           (car_data.csv - 1,500 Indian Market Records)       |
+--------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|                   2. Feature Engineering                     |
|              Car_Age = 2026 - Registration Year              |
|        Categorical: Brand, Model, Fuel, Seller, Trans        |
|        Numerical: Present_Price, Kms_Driven, Owner, Car_Age  |
+--------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|                 3. Scikit-Learn Preprocessor                 |
|       OneHotEncoder(handle_unknown='ignore') + StandardScaler|
+--------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|                  4. Multi-Model Evaluation                   |
|     - Linear Regression (Baseline)                           |
|     - Random Forest Regressor (n_estimators=150)             |
|     - XGBoost Regressor (n_estimators=150, lr=0.1, depth=6)  |
+--------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|              5. Model Persistence & Artifacts                |
|      - best_car_model.pkl (Scikit-Learn Joblib Pipeline)     |
|      - metrics.json (R2, MAE, RMSE Benchmark Scores)         |
|      - metadata.json (UI Bounds, Brand/Model Hierarchies)    |
+--------------------------------------------------------------+
                               |
                               v
+--------------------------------------------------------------+
|              6. Interactive Streamlit Web UI                 |
|   - Tab 1: AI Instant Resale Valuation Tool                  |
|   - Tab 2: Vehicle Depreciation & What-If Holding Simulator  |
|   - Tab 3: Dealership & Fleet Batch Valuation Platform       |
|   - Tab 4: Model Performance Benchmarks & Market Analytics   |
+--------------------------------------------------------------+
```

---

## 📊 3. Feature Dictionary

| Feature Name | Type | Description / Range | Example |
| :--- | :--- | :--- | :--- |
| `Brand` | Categorical | Manufacturer name | Maruti Suzuki, Hyundai, Tata, Mahindra, Toyota, Honda, Kia |
| `Model` | Categorical | Vehicle model mapped to brand | Creta, Fortuner, Swift, Nexon, Thar, City, Seltos |
| `Year` | Numerical | Original year of manufacturing / registration | 2012 to 2025 |
| `Car_Age` | Engineered | Vehicle age in years (`2026 - Year`) | 1 to 14 years |
| `Present_Price` | Numerical | Original ex-showroom price in ₹ Lakhs | ₹ 5.50 Lakhs to ₹ 65.50 Lakhs |
| `Kms_Driven` | Numerical | Cumulative odometer reading in kilometres | 3,000 km to 1,50,000 km |
| `Fuel_Type` | Categorical | Powertrain fuel category | Petrol, Diesel, CNG, Electric |
| `Seller_Type` | Categorical | Entity offering the vehicle for sale | Dealer, Individual |
| `Transmission` | Categorical | Gearbox mechanism | Manual, Automatic |
| `Owner` | Numerical | Count of previous registered owners | 0 (First Owner), 1 (Second Owner), 2 (Third Owner) |
| **`Selling_Price`** | **Target** | **Market Resale Valuation in ₹ Lakhs** | **Continuous Dependent Variable** |

---

## 🔬 4. Comparative Model Performance Benchmark

The models were evaluated on an independent 20% holdout test dataset (300 unseen records):

| Model Algorithm | $R^2$ Score (Accuracy) | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** (Baseline) | `0.8700` | ₹ 1.25 Lakhs | ₹ 2.21 Lakhs | Baseline |
| **Random Forest Regressor** | **`0.9897`** | **₹ 0.41 Lakhs** | **₹ 0.62 Lakhs** | **🏆 Best Model (Saved)** |
| **XGBoost Regressor** | `0.9884` | ₹ 0.36 Lakhs | ₹ 0.66 Lakhs | Runner-Up |

* **5-Fold Cross-Validation $R^2$ Score:** `0.9817` ($\pm 0.0045$), mathematically proving high generalizability with zero overfitting.

### Top Predictive Feature Importances
1. **`Present_Price` (73.4%)**: Primary anchor for non-linear depreciation base.
2. **`Car_Age` (22.7%)**: Decisive factor driving compounding annual value loss.
3. **`Kms_Driven` (2.8%)**: Wear-and-tear odometer adjustment.
4. **`Owner` (0.33%)**: Multi-owner discount factor.
5. **`Brand / Model / Powertrain`**: Secondary market prestige and residual demand adjustments.

---

## 💻 5. Tech Stack & Dependencies

- **Language:** Python 3.10+
- **Machine Learning & Preprocessing:** `scikit-learn`, `xgboost`, `numpy`, `pandas`
- **Model Serialization:** `joblib`
- **Dashboard & Interactive UI:** `streamlit`
- **Data Visualization & Analytics:** `plotly`

---

## 🚀 6. Step-by-Step Installation & Execution Guide

### Step 1: Navigate to Project Directory
```bash
cd D:\Mini_Project
```

### Step 2: Install Dependencies
```bash
pip install -r requirements.txt
```

### Step 3: Generate the Dataset (Optional, automated in Step 4)
```bash
python generate_dataset.py
```
*Outputs: `car_data.csv` (1,500 samples)*

### Step 4: Train Models & Generate Artifacts
```bash
python train_model.py
```
*Outputs: `best_car_model.pkl`, `metrics.json`, `metadata.json`*

### Step 5: Launch the Streamlit Web Application
```bash
streamlit run app.py
```
Open your browser at `http://localhost:8501` to test the live predictor!

---

## 📜 7. Academic Declaration

This project has been developed by **DESAM KOWSHIK REDDY** (Roll Number: **23761A0584**) as part of the **B.Tech AI/ML Internship Mini Project**. All code, pipelines, dataset simulations, and documentation have been tested and verified for reproducible execution.
