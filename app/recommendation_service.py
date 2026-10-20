
from app.data_service import get_properties


def recommend_properties(
    city=None,
    budget=50_000_000,
    bedrooms=None,
    property_type=None,
    limit=10,
):
    df = get_properties().copy()

    df = df[df["price_pkr"] <= budget]

    if city:
        df = df[df["city"].str.lower() == city.lower()]

    if bedrooms is not None:
        df = df[df["bedrooms"] >= bedrooms]

    if property_type:
        df = df[
            df["property_type"].str.lower() == property_type.lower()
        ]

    df = df.sort_values("price_pkr").head(limit)

    return df.to_dict(orient="records")
