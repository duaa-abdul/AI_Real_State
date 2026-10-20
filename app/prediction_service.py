from pathlib import Path

import joblib
import pandas as pd

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder

from app.data_service import get_properties

BASE_DIR = Path(__file__).resolve().parent.parent
MODEL_DIR = BASE_DIR / "models"
MODEL_FILE = MODEL_DIR / "house_price_model.pkl"

FEATURES = [
    "city",
    "property_type",
    "area_sqft",
    "bedrooms",
    "bathrooms",
    "age_years",
]

CATEGORICAL = ["city", "property_type"]
NUMERIC = ["area_sqft", "bedrooms", "bathrooms", "age_years"]


def train_model():
    df = get_properties()

    X = df[FEATURES]
    y = df["price_pkr"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessing = ColumnTransformer([
        (
            "categories",
            OneHotEncoder(handle_unknown="ignore"),
            CATEGORICAL,
        ),
        ("numbers", "passthrough", NUMERIC),
    ])

    model = Pipeline([
        ("preprocessing", preprocessing),
        (
            "regressor",
            RandomForestRegressor(
                n_estimators=150,
                random_state=42,
                min_samples_leaf=2,
            ),
        ),
    ])

    model.fit(X_train, y_train)

    predictions = model.predict(X_test)

    metrics = {
        "mae_pkr": round(float(mean_absolute_error(y_test, predictions))),
        "r2": round(float(r2_score(y_test, predictions)), 4),
    }

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, MODEL_FILE)

    return metrics


def load_model():
    if not MODEL_FILE.exists():
        train_model()

    return joblib.load(MODEL_FILE)


def predict_price(property_data):
    model = load_model()

    row = pd.DataFrame([{
        "city": property_data["city"],
        "property_type": property_data["property_type"],
        "area_sqft": property_data["area_sqft"],
        "bedrooms": property_data["bedrooms"],
        "bathrooms": property_data["bathrooms"],
        "age_years": property_data["age_years"],
    }])

    prediction = float(model.predict(row)[0])

    return {
        "estimated_price_pkr": round(max(0, prediction)),
        "currency": "PKR",
        "disclaimer": (
            "Estimated value from a synthetic demonstration dataset; "
            "not a verified market valuation."
        ),
    }
