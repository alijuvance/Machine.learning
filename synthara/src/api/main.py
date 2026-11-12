"""
Synthara — FastAPI Application
REST API exposing synthetic data generation for African financial data.

Endpoints:
    GET  /api/v1/health          — Health check
    POST /api/v1/analyze         — Analyze a dataset's patterns
    POST /api/v1/generate        — Generate synthetic data from uploaded CSV
    GET  /api/v1/job/{job_id}    — Check job status / retrieve results
    GET  /api/v1/reports         — List available validation reports
"""

import io
import logging
from contextlib import asynccontextmanager
from datetime import datetime

import pandas as pd
from fastapi import FastAPI, File, HTTPException, Query, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse

from .schemas import (
    AnalyzeRequest,
    GenerateRequest,
    HealthResponse,
    JobResponse,
    JobStatus,
    ModelName,
    ProfileResponse,
)
from .services import SyntharaService, DATA_GENERATED, DATA_REPORTS

logger = logging.getLogger(__name__)

# ─── App Lifecycle ────────────────────────────────────────────────

service = SyntharaService()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application startup and shutdown events."""
    logger.info("🌍 Synthara API starting...")
    yield
    logger.info("🌍 Synthara API shutting down.")


# ─── App Creation ─────────────────────────────────────────────────

app = FastAPI(
    title="Synthara API",
    description=(
        "🌍 Synthetic Data Engine for African Financial AI. "
        "Generate statistically faithful datasets calibrated on mobile money, "
        "microfinance, and informal economy patterns."
    ),
    version="0.1.0",
    lifespan=lifespan,
)

# CORS — allow Next.js frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173", "*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Routes ───────────────────────────────────────────────────────

@app.get("/api/v1/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """Check if the API is healthy and responding."""
    return HealthResponse(
        status="healthy",
        version="0.1.0",
        timestamp=datetime.now(),
    )


@app.post("/api/v1/analyze", tags=["Analysis"])
async def analyze_dataset(
    file: UploadFile = File(..., description="CSV file to analyze"),
    detect_african_patterns: bool = Query(
        True, description="Run African-specific pattern detection"
    ),
    max_rows: int = Query(None, description="Max rows to load (for large files)"),
):
    """
    Upload a CSV dataset and receive a statistical profile
    with African-specific pattern analysis.
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content), nrows=max_rows)

        result = service.analyze_dataset(
            df,
            name=file.filename,
            detect_african=detect_african_patterns,
        )
        return JSONResponse(content=result)

    except Exception as e:
        logger.error(f"Analysis failed: {e}", exc_info=True)
        raise HTTPException(500, f"Analysis failed: {str(e)}")


@app.post("/api/v1/generate", tags=["Generation"])
async def generate_synthetic_data(
    file: UploadFile = File(..., description="Source CSV file"),
    model_name: ModelName = Query(ModelName.ctgan, description="Generation model"),
    num_rows: int = Query(10000, ge=100, le=1_000_000, description="Rows to generate"),
    apply_constraints: bool = Query(True, description="Apply African business constraints"),
):
    """
    Upload a source CSV and receive synthetic data back.
    Also generates a validation report (fidelity + privacy).
    """
    if not file.filename.endswith(".csv"):
        raise HTTPException(400, "Only CSV files are supported")

    try:
        content = await file.read()
        df = pd.read_csv(io.BytesIO(content))

        # Save uploaded file
        service.save_uploaded_file(content, file.filename)

        # Generate
        synthetic_df, report = service.generate_synthetic(
            source_df=df,
            model_name=model_name.value,
            num_rows=num_rows,
            apply_constraints=apply_constraints,
        )

        # Save synthetic CSV
        csv_path = service.save_synthetic_csv(synthetic_df, model_name.value)

        # Return CSV as download
        csv_buffer = io.StringIO()
        synthetic_df.to_csv(csv_buffer, index=False)
        csv_buffer.seek(0)

        return StreamingResponse(
            io.BytesIO(csv_buffer.getvalue().encode()),
            media_type="text/csv",
            headers={
                "Content-Disposition": f"attachment; filename=synthetic_{model_name.value}.csv",
                "X-Synthara-Fidelity": str(round(report["overall"]["fidelity_score"], 4)),
                "X-Synthara-Privacy": str(round(report["overall"]["privacy_score"], 4)),
                "X-Synthara-Verdict": report["overall"]["verdict"],
                "X-Synthara-Rows": str(len(synthetic_df)),
            },
        )

    except Exception as e:
        logger.error(f"Generation failed: {e}", exc_info=True)
        raise HTTPException(500, f"Generation failed: {str(e)}")


@app.get("/api/v1/reports", tags=["Reports"])
async def list_reports():
    """List all available validation and benchmark reports."""
    reports = []
    for f in DATA_REPORTS.glob("*.json"):
        reports.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "created_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    reports.sort(key=lambda x: x["created_at"], reverse=True)
    return {"reports": reports, "count": len(reports)}


@app.get("/api/v1/reports/{filename}", tags=["Reports"])
async def get_report(filename: str):
    """Download a specific validation/benchmark report."""
    filepath = DATA_REPORTS / filename
    if not filepath.exists():
        raise HTTPException(404, f"Report not found: {filename}")
    return FileResponse(filepath, media_type="application/json")


@app.get("/api/v1/generated", tags=["Data"])
async def list_generated_datasets():
    """List all generated synthetic datasets."""
    datasets = []
    for f in DATA_GENERATED.glob("*.csv"):
        datasets.append({
            "filename": f.name,
            "size_bytes": f.stat().st_size,
            "created_at": datetime.fromtimestamp(f.stat().st_mtime).isoformat(),
        })
    datasets.sort(key=lambda x: x["created_at"], reverse=True)
    return {"datasets": datasets, "count": len(datasets)}


@app.get("/api/v1/generated/{filename}", tags=["Data"])
async def download_generated_dataset(filename: str):
    """Download a generated synthetic dataset."""
    filepath = DATA_GENERATED / filename
    if not filepath.exists():
        raise HTTPException(404, f"Dataset not found: {filename}")
    return FileResponse(filepath, media_type="text/csv")


# ─── Entry Point ──────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    )
    uvicorn.run(app, host="0.0.0.0", port=8000)
