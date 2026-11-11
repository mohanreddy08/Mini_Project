# B.TECH AI/ML INTERNSHIP MINI PROJECT REPORT

---

## STUDENT & PROJECT METADATA
* **Project Title:** Used Car Price Prediction System (Indian Market)
* **Candidate Name:** DESAM KOWSHIK REDDY
* **University Roll Number:** 23761A0584
* **Department:** Computer Science & Engineering (Artificial Intelligence & Machine Learning)
* **Domain:** Supervised Tabular Machine Learning & Predictive Modeling
* **Target Currency:** Indian Rupees (₹ in Lakhs)
* **Academic Year:** 2026

---

## CERTIFICATE OF AUTHENTICITY

This is to certify that the project report titled **"Used Car Price Prediction System (Indian Market)"** submitted by **DESAM KOWSHIK REDDY (Roll Number: 23761A0584)** in partial fulfillment of the requirements for the award of the degree of **Bachelor of Technology in Artificial Intelligence & Machine Learning** is a bona fide record of work carried out independently.

---

## ABSTRACT

The pre-owned automobile industry in India has transitioned into an organized, high-volume market driven by consumer demand and rapid vehicle refresh cycles. However, fair pricing remains challenging due to non-linear depreciation patterns influenced by vehicle age, odometer mileage, brand retention dynamics, fuel preferences, and transmission types.

This project designs, develops, and deploys an automated Machine Learning pipeline for pre-owned car valuation in the Indian automotive market. A comprehensive 1,500-sample dataset modeled after real-world CarDekho listings was curated across 7 leading manufacturers (Maruti Suzuki, Hyundai, Tata, Mahindra, Honda, Toyota, and Kia) spanning diverse vehicle segments.

The data processing pipeline applies Scikit-Learn `ColumnTransformer` to enforce categorical one-hot encoding alongside numerical feature standardization. Three machine learning regression algorithms were trained and benchmarked: **Linear Regression (Baseline)**, **Random Forest Regressor**, and **Extreme Gradient Boosting (XGBoost Regressor)**. 

The **Random Forest Regressor** demonstrated superior predictive performance, achieving an **$R^2$ Score of 0.9897** on unseen test data with a **Mean Absolute Error (MAE) of ₹ 0.4072 Lakhs**, validated through 5-fold cross-validation ($R^2 = 0.9817 \pm 0.0045$). The model is deployed via a Streamlit web application providing single-vehicle appraisal with fair price brackets ($\pm 5\%$), future depreciation simulation, institutional fleet batch processing, and interactive explainability dashboards.

---

## CHAPTER 1: INTRODUCTION & PROBLEM STATEMENT

### 1.1 Background
The Indian pre-owned car market has expanded rapidly, driven by certified dealership platforms. However, asymmetric information between private buyers, sellers, and dealers often results in inconsistent valuations.

### 1.2 Non-Linear Depreciation Dynamics
Automotive depreciation in India exhibits distinct characteristics:
1. **Initial Cliff:** Year one depreciation drops by approximately 12–15% upon showroom exit.
2. **Compounding Decay:** Subsequent years experience 6–8% compounding depreciation.
3. **Brand Retention:** Brands with high spare parts availability and reliability ratings (such as Toyota and Maruti Suzuki) hold residual value substantially better than competitors.
4. **Mileage & Powertrain Factors:** Diesel powertrains and automatic gearboxes retain higher demand in SUV and crossover categories.

### 1.3 Project Objectives
- Synthesize an authentic multi-brand Indian automotive dataset.
- Implement an automated preprocessing and model comparison pipeline.
- Benchmark Linear Regression, Random Forest, and XGBoost regressor architectures.
- Deploy an intuitive web application supporting individual and institutional appraisal workflows.

---

## CHAPTER 2: SYSTEM ARCHITECTURE & WORKFLOW

```
[Raw Car Listings: 1,500 Samples]
               │
               ▼
   [Feature Engineering: Car_Age = 2026 - Year]
               │
               ▼
   [Scikit-Learn ColumnTransformer]
   ├── Categorical (Brand, Model, Fuel, Seller, Trans): OneHotEncoder
   └── Numerical (Present_Price, Kms, Owner, Age): StandardScaler
               │
       ┌───────┼───────┐
       ▼       ▼       ▼
 [LinearReg] [RandomForest] [XGBoost]
       │       │       │
       └───────┼───────┘
               ▼
   [Benchmark & Validation: 5-Fold CV]
               │
               ▼
   [Export: best_car_model.pkl & metrics.json]
               │
               ▼
   [Interactive Streamlit Dashboard]
   ├── Tab 1: Instant Resale Price Predictor
   ├── Tab 2: Depreciation & What-If Simulator
   ├── Tab 3: Dealership Fleet Batch Valuation
   └── Tab 4: Performance & Market EDA
```

---

## CHAPTER 3: DATASET SPECIFICATIONS & FEATURE IMPORTANCE

### 3.1 Input Features
- `Brand`: Nominal categorical (Maruti Suzuki, Hyundai, Tata, Mahindra, Honda, Toyota, Kia)
- `Model`: Nominal categorical mapped to each brand (Swift, Creta, Fortuner, Nexon, City, etc.)
- `Present_Price`: Original ex-showroom price (₹ 5.50 to ₹ 65.50 Lakhs)
- `Kms_Driven`: Odometer mileage (3,000 to 1,50,000 km)
- `Fuel_Type`: Petrol, Diesel, CNG, Electric
- `Seller_Type`: Dealer, Individual
- `Transmission`: Manual, Automatic
- `Owner`: 0 (First Owner), 1 (Second Owner), 2 (Third Owner)
- `Car_Age`: Elapsed age calculated as `2026 - Year`

### 3.2 Feature Importance Ranking
1. **`Present_Price` (73.4%)**: Primary base asset anchor.
2. **`Car_Age` (22.7%)**: Main driver of compounding depreciation.
3. **`Kms_Driven` (2.8%)**: Mechanical wear-and-tear penalty.
4. **`Owner` (0.33%)**: Ownership transfer discount.
5. **`Brand / Fuel / Transmission` (< 1%)**: Brand prestige and powertrain premiums.

---

## CHAPTER 4: EXPERIMENTAL RESULTS & EVALUATION

The models were evaluated on an independent 20% holdout test dataset (300 unseen records):

| Model Architecture | $R^2$ Score (Accuracy) | Mean Absolute Error (MAE) | Root Mean Squared Error (RMSE) | Status |
| :--- | :---: | :---: | :---: | :---: |
| **Linear Regression** (Baseline) | `0.8700` | ₹ 1.2497 Lakhs | ₹ 2.2117 Lakhs | Baseline |
| **Random Forest Regressor** | **`0.9897`** | **₹ 0.4072 Lakhs** | **₹ 0.6229 Lakhs** | **🏆 Best Model (Saved)** |
| **XGBoost Regressor** | `0.9884` | ₹ 0.3559 Lakhs | ₹ 0.6595 Lakhs | Competitive Alternative |

* **5-Fold Cross-Validation:** Achieved an $R^2$ of **0.9817 ($\pm 0.0045$)**, proving minimal variance and robust stability across distinct data splits.

---

## CHAPTER 5: CONCLUSION & FUTURE ENHANCEMENTS

### 5.1 Conclusion
The project successfully established that ensemble tree-based models (Random Forest and XGBoost) significantly outperform linear approximations when modeling automotive residual depreciation, delivering sub-0.5 Lakh prediction error across varied price bands.

### 5.2 Future Scope
- Integration with live RTO database APIs via vehicle registration number lookup.
- Computer vision module for vehicle exterior dent and scratch damage estimation.
- City-specific regional demand adjustments (e.g., Delhi-NCR 10-year diesel rule).

---

**Report Authored by:** DESAM KOWSHIK REDDY (23761A0584)  
**Workspace:** `D:\Mini_Project`
