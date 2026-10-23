"""FastAPI interface for the existing job-posting risk engine."""
from pathlib import Path
from typing import Any

from fastapi import FastAPI, HTTPException
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field, model_validator

from src.config import HIGH_RISK_THRESHOLD, LOW_RISK_THRESHOLD
from src.predict import get_prediction_engine


app = FastAPI(title="Job Posting Risk Analyzer API", version="1.0.0")
FRONTEND_DIST = Path(__file__).parent / "frontend" / "dist"


class JobPosting(BaseModel):
    title: str = Field(default="", max_length=500)
    company_profile: str = Field(default="", max_length=10000)
    description: str = Field(default="", max_length=20000)
    requirements: str = Field(default="", max_length=10000)
    benefits: str = Field(default="", max_length=10000)
    telecommuting: bool = False
    has_company_logo: bool = False
    has_questions: bool = False
    employment_type: str = "Unknown"
    required_experience: str = "Unknown"
    required_education: str = "Unknown"
    salary_range: str = ""
    model_name: str = "Linear SVM"
    low_threshold: float = Field(default=LOW_RISK_THRESHOLD, ge=0.0, le=1.0)
    high_threshold: float = Field(default=HIGH_RISK_THRESHOLD, ge=0.0, le=1.0)

    @model_validator(mode="after")
    def thresholds_are_ordered(self):
        if self.low_threshold >= self.high_threshold:
            raise ValueError("The moderate-risk threshold must be below the high-risk threshold.")
        return self


@app.get("/api/health")
def health() -> dict[str, Any]:
    engine = get_prediction_engine()
    return {"ready": engine.is_ready, "models": engine.get_available_models()}


@app.post("/api/predict")
def predict(posting: JobPosting) -> dict[str, Any]:
    engine = get_prediction_engine()
    if not engine.is_ready:
        raise HTTPException(status_code=503, detail="Model artifacts are unavailable. Run python train_evaluate.py first.")

    payload = posting.model_dump()
    model_name = payload.pop("model_name")
    low_threshold = payload.pop("low_threshold")
    high_threshold = payload.pop("high_threshold")
    for field in ("telecommuting", "has_company_logo", "has_questions"):
        payload[field] = int(payload[field])

    try:
        result = engine.predict(
            payload,
            model_name=model_name,
            low_thresh=low_threshold,
            high_thresh=high_threshold,
        )
    except (TypeError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error

    result.pop("highlighted_html", None)
    return result


if FRONTEND_DIST.exists():
    assets_dir = FRONTEND_DIST / "assets"
    if assets_dir.exists():
        app.mount("/assets", StaticFiles(directory=assets_dir), name="assets")

    @app.get("/{frontend_path:path}", include_in_schema=False)
    def frontend(frontend_path: str):
        requested_file = (FRONTEND_DIST / frontend_path).resolve()
        if requested_file.is_relative_to(FRONTEND_DIST.resolve()) and requested_file.is_file():
            return FileResponse(requested_file)
        return FileResponse(FRONTEND_DIST / "index.html")