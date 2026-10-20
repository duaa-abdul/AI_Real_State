
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
DATA_FILE = DATA_DIR / "properties.csv"

CITIES = ["Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad"]
PROPERTY_TYPES = ["House", "Apartment", "Villa"]


def generate_dataset():
    """Generate a reproducible sample property dataset."""
    DATA_DIR.mkdir(parents=True, exist_ok=True)

    if DATA_FILE.exists():
        return pd.read_csv(DATA_FILE)

    rng = np.random.default_rng(42)
    rows = []

    city_base = {
        "Karachi": 6500,
        "Lahore": 6000,
        "Islamabad": 8500,
        "Rawalpindi": 4500,
        "Faisalabad": 3500,
    }

    for i in range(300):
        city = CITIES[i % len(CITIES)]
        property_type = PROPERTY_TYPES[i % len(PROPERTY_TYPES)]
        area = int(rng.integers(500, 4501))
        bedrooms = int(rng.integers(1, 7))
        bathrooms = int(rng.integers(1, 6))
        age = int(rng.integers(0, 31))

        type_multiplier = {
            "House": 1.0,
            "Apartment": 0.82,
            "Villa": 1.35,
        }[property_type]

        price = (
            area * city_base[city]
            + bedrooms * 1_500_000
            + bathrooms * 900_000
            - age * 120_000
        ) * type_multiplier

        price += rng.normal(0, 2_000_000)
        price = max(1_500_000, int(price))

        rows.append({
            "property_id": i + 1,
            "city": city,
            "property_type": property_type,
            "area_sqft": area,
            "bedrooms": bedrooms,
            "bathrooms": bathrooms,
            "age_years": age,
            "price_pkr": price,
        })

    df = pd.DataFrame(rows)
    df.to_csv(DATA_FILE, index=False)
    return df


def get_properties():
    return generate_dataset()


def get_statistics():
    df = get_properties()

    return {
        "total_properties": int(len(df)),
        "average_price": round(float(df["price_pkr"].mean())),
        "minimum_price": int(df["price_pkr"].min()),
        "maximum_price": int(df["price_pkr"].max()),
        "average_area": round(float(df["area_sqft"].mean())),
        "city_counts": {
            str(city): int(count)
            for city, count in df["city"].value_counts().items()
        },
        "average_prices_by_city": {
            str(city): round(float(price))
            for city, price in
            df.groupby("city")["price_pkr"].mean().items()
        },
    }
