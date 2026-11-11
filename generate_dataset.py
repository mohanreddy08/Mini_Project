"""
Realistic Indian Used Car Dataset Generator
Modeled after CarDekho / Indian Market Listings
Student: DESAM KOWSHIK REDDY (23761A0584)
Domain: B.Tech AI/ML Internship Mini Project
"""

import os
import numpy as np
import pandas as pd

# Set random seed for reproducibility
np.random.seed(42)

BRAND_MODELS = {
    "Maruti Suzuki": {
        "Swift": (6.0, 9.5, ["Petrol", "CNG"]),
        "Baleno": (6.6, 10.2, ["Petrol", "CNG"]),
        "Dzire": (6.5, 9.8, ["Petrol", "CNG"]),
        "Brezza": (8.3, 14.2, ["Petrol", "CNG"]),
        "Ertiga": (8.8, 13.5, ["Petrol", "CNG"]),
        "Wagon R": (5.5, 7.5, ["Petrol", "CNG"]),
    },
    "Hyundai": {
        "Creta": (11.0, 20.2, ["Petrol", "Diesel"]),
        "i20": (7.0, 11.8, ["Petrol", "Diesel"]),
        "Venue": (8.0, 13.5, ["Petrol", "Diesel"]),
        "Verna": (11.0, 17.5, ["Petrol", "Diesel"]),
        "Grand i10": (5.9, 8.8, ["Petrol", "CNG"]),
        "Tucson": (29.0, 35.8, ["Petrol", "Diesel"]),
    },
    "Tata": {
        "Nexon": (8.0, 15.5, ["Petrol", "Diesel", "Electric"]),
        "Harrier": (15.5, 25.0, ["Diesel"]),
        "Safari": (16.2, 26.5, ["Diesel"]),
        "Altroz": (6.6, 11.0, ["Petrol", "Diesel", "CNG"]),
        "Punch": (6.1, 10.3, ["Petrol", "CNG"]),
        "Tiago": (5.6, 8.6, ["Petrol", "CNG", "Electric"]),
    },
    "Mahindra": {
        "Thar": (11.5, 17.8, ["Petrol", "Diesel"]),
        "XUV700": (14.5, 26.8, ["Petrol", "Diesel"]),
        "Scorpio-N": (13.8, 24.5, ["Petrol", "Diesel"]),
        "Bolero": (9.8, 11.2, ["Diesel"]),
        "XUV300": (8.5, 14.8, ["Petrol", "Diesel"]),
    },
    "Honda": {
        "City": (12.0, 16.5, ["Petrol"]),
        "Amaze": (7.2, 10.0, ["Petrol", "Diesel"]),
        "Elevate": (11.8, 16.5, ["Petrol"]),
        "WR-V": (8.8, 12.5, ["Petrol", "Diesel"]),
        "Civic": (18.0, 23.0, ["Petrol", "Diesel"]),
    },
    "Toyota": {
        "Fortuner": (35.0, 51.5, ["Petrol", "Diesel"]),
        "Innova Crysta": (20.0, 29.5, ["Petrol", "Diesel"]),
        "Urban Cruiser": (9.2, 13.5, ["Petrol"]),
        "Glanza": (6.8, 10.2, ["Petrol", "CNG"]),
        "Camry": (42.0, 48.0, ["Petrol"]),
    },
    "Kia": {
        "Seltos": (11.0, 20.5, ["Petrol", "Diesel"]),
        "Sonet": (8.0, 14.8, ["Petrol", "Diesel"]),
        "Carens": (10.5, 19.8, ["Petrol", "Diesel"]),
        "Carnival": (31.0, 36.5, ["Diesel"]),
        "EV6": (60.5, 65.9, ["Electric"]),
    },
}

def generate_car_dataset(n_samples: int = 1500, output_path: str = "car_data.csv") -> pd.DataFrame:
    """Generates realistic Indian market used car records and writes to CSV."""
    brands = list(BRAND_MODELS.keys())
    # Brand market share distribution in India
    brand_weights = [0.28, 0.22, 0.16, 0.14, 0.08, 0.07, 0.05]

    data = []
    current_year = 2026

    for _ in range(n_samples):
        brand = np.random.choice(brands, p=brand_weights)
        models = list(BRAND_MODELS[brand].keys())
        model = np.random.choice(models)

        min_p, max_p, allowed_fuels = BRAND_MODELS[brand][model]
        present_price = round(np.random.uniform(min_p, max_p), 2)

        # Registration Year from 2012 to 2025
        year_probs = np.linspace(0.04, 0.12, 14)
        year_probs /= year_probs.sum()
        year = int(np.random.choice(np.arange(2012, 2026), p=year_probs))
        age = current_year - year

        # Kms Driven correlated with age
        annual_mileage = np.random.normal(11000, 2800)
        annual_mileage = max(3500, annual_mileage)
        kms_driven = int(age * annual_mileage + np.random.uniform(-2000, 4000))
        kms_driven = int(np.clip(kms_driven, 3000, 150000))

        # Fuel type
        fuel = np.random.choice(allowed_fuels)

        # Transmission (higher probability of automatic on premium/EV cars)
        if fuel == "Electric" or present_price > 25.0:
            transmission = np.random.choice(["Manual", "Automatic"], p=[0.15, 0.85])
        elif present_price > 14.0:
            transmission = np.random.choice(["Manual", "Automatic"], p=[0.55, 0.45])
        else:
            transmission = np.random.choice(["Manual", "Automatic"], p=[0.80, 0.20])

        # Seller type
        seller_type = np.random.choice(["Dealer", "Individual"], p=[0.62, 0.38])

        # Owners: 0 (1st Owner), 1 (2nd Owner), 2 (3rd Owner)
        if age <= 2:
            owner = 0
        elif age <= 5:
            owner = int(np.random.choice([0, 1], p=[0.82, 0.18]))
        elif age <= 8:
            owner = int(np.random.choice([0, 1, 2], p=[0.60, 0.32, 0.08]))
        else:
            owner = int(np.random.choice([0, 1, 2], p=[0.42, 0.42, 0.16]))

        # Calculate Realistic Selling_Price (INR Lakhs)
        # Non-linear compounding depreciation formula
        base_retention = 0.86 * (0.932 ** max(0, age - 1))

        # Brand resilience in Indian resale market
        brand_retention_factor = {
            "Toyota": 1.05,
            "Maruti Suzuki": 1.03,
            "Hyundai": 1.00,
            "Honda": 0.99,
            "Kia": 0.98,
            "Tata": 0.97,
            "Mahindra": 0.99
        }.get(brand, 1.0)

        # Mileage penalty (~4% per 50,000 km)
        mileage_penalty = (kms_driven / 50000.0) * 0.04

        # Fuel modifier
        fuel_mod = 0.0
        if fuel == "Diesel":
            fuel_mod = 0.02
        elif fuel == "CNG":
            fuel_mod = -0.02
        elif fuel == "Electric":
            fuel_mod = -0.03 if age > 4 else 0.01

        # Transmission modifier (automatic resale premium)
        trans_mod = 0.025 if transmission == "Automatic" else 0.0

        # Seller modifier (dealer warranty markup)
        seller_mod = 0.02 if seller_type == "Dealer" else -0.01

        # Owner penalty
        owner_penalty = owner * 0.06

        net_retention = (base_retention * brand_retention_factor) - mileage_penalty + fuel_mod + trans_mod + seller_mod - owner_penalty
        net_retention = np.clip(net_retention, 0.16, 0.93)

        # Selling price with small gaussian noise
        calculated_price = present_price * net_retention + np.random.normal(0, 0.18)

        # Guardrails: Min price 1.0 Lakh, max price 94% of present price
        selling_price = float(np.clip(calculated_price, 1.0, present_price * 0.94))
        selling_price = round(selling_price, 2)

        data.append({
            "Brand": brand,
            "Model": model,
            "Year": year,
            "Selling_Price": selling_price,
            "Present_Price": present_price,
            "Kms_Driven": kms_driven,
            "Fuel_Type": fuel,
            "Seller_Type": seller_type,
            "Transmission": transmission,
            "Owner": owner,
        })

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Successfully generated {len(df)} car records at '{output_path}'.")
    return df

if __name__ == "__main__":
    generate_car_dataset(1500, "car_data.csv")
