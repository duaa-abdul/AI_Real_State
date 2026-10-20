from contextlib import asynccontextmanager
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.responses import HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.templating import Jinja2Templates
from fastapi import Request
from pydantic import BaseModel, Field

from app.data_service import get_properties, get_statistics
from app.prediction_service import train_model, predict_price
from app.recommendation_service import recommend_properties
from app.rag_service import initialize_rag
from app.ai_service import ask_real_estate_assistant

BASE_DIR = Path(__file__).resolve().parent


@asynccontextmanager
async def lifespan(app: FastAPI):
    get_properties()
    train_model()
    initialize_rag()
    yield


app = FastAPI(
    title="AI Real Estate Assistant",
    description="Property analytics, price prediction, recommendations and RAG.",
    version="1.0.0",
    lifespan=lifespan,
)

app.mount(
    "/static",
    StaticFiles(directory=str(BASE_DIR / "static")),
    name="static",
)

templates = Jinja2Templates(directory=str(BASE_DIR / "templates"))


class PredictionRequest(BaseModel):
    city: str
    property_type: str
    area_sqft: int = Field(gt=0, le=100000)
    bedrooms: int = Field(ge=1, le=30)
    bathrooms: int = Field(ge=1, le=30)
    age_years: int = Field(ge=0, le=200)


class RecommendationRequest(BaseModel):
    city: str | None = None
    budget: int = Field(default=50_000_000, gt=0)
    bedrooms: int | None = Field(default=None, ge=1, le=30)
    property_type: str | None = None
    limit: int = Field(default=10, ge=1, le=50)


class ChatRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1500)


@app.get("/", response_class=HTMLResponse)
def home(request: Request):
    return templates.TemplateResponse(
        request=request,
        name="index.html",
        context={},
    )


@app.get("/api/health")
def health():
    return {"status": "ok", "application": "AI Real Estate Assistant"}


@app.get("/api/stats")
def stats():
    return get_statistics()


@app.get("/api/properties")
def properties(
    city: str | None = None,
    property_type: str | None = None,
    limit: int = Query(default=100, ge=1, le=300),
):
    df = get_properties().copy()

    if city:
        df = df[df["city"].str.lower() == city.lower()]

    if property_type:
        df = df[
            df["property_type"].str.lower() == property_type.lower()
        ]

    return df.head(limit).to_dict(orient="records")


@app.get("/api/analysis")
def analysis():
    df = get_properties()

    return {
        "statistics": get_statistics(),
        "properties_by_type": {
            str(key): int(value)
            for key, value in df["property_type"].value_counts().items()
        },
        "bedroom_counts": {
            str(key): int(value)
            for key, value in df["bedrooms"].value_counts().sort_index().items()
        },
    }


@app.post("/api/predict")
def predict(payload: PredictionRequest):
    valid_cities = {
        "Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad"
    }
    valid_types = {"House", "Apartment", "Villa"}

    if payload.city not in valid_cities:
        raise HTTPException(status_code=400, detail="Select a supported city.")

    if payload.property_type not in valid_types:
        raise HTTPException(status_code=400, detail="Invalid property type.")

    try:
        return predict_price(payload.model_dump())
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Prediction failed: {exc}",
        ) from exc


@app.post("/api/recommend")
def recommend(payload: RecommendationRequest):
    valid_cities = {
        "Karachi", "Lahore", "Islamabad", "Rawalpindi", "Faisalabad"
    }
    valid_types = {"House", "Apartment", "Villa"}

    if payload.city and payload.city not in valid_cities:
        raise HTTPException(status_code=400, detail="Select a supported city.")

    if payload.property_type and payload.property_type not in valid_types:
        raise HTTPException(status_code=400, detail="Invalid property type.")

    return {
        "properties": recommend_properties(
            city=payload.city,
            budget=payload.budget,
            bedrooms=payload.bedrooms,
            property_type=payload.property_type,
            limit=payload.limit,
        )
    }


@app.post("/api/chat")
def chat(payload: ChatRequest):
    try:
        return ask_real_estate_assistant(payload.question)
    except RuntimeError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except Exception:
        raise HTTPException(
            status_code=502,
            detail="AI service is unavailable. Check your Groq key, model and internet connection.",
        )


@app.post("/api/rag/rebuild")
def rebuild_knowledge_base():
    try:
        return initialize_rag()
    except Exception as exc:
        raise HTTPException(
            status_code=500,
            detail=f"Knowledge base build failed: {exc}",
        ) from exc
