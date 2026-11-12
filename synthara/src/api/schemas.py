"""
Synthara — API Pydantic Schemas
Request/response models for the FastAPI endpoints.
"""

from datetime import datetime
from enum import Enum
from typing import Optional

from pydantic import BaseModel, Field


class ModelName(str, Enum):
    """Available synthetic data generation models."""
    ctgan = "ctgan"
    gaussian_copula = "gaussian_copula"
    tvae = "tvae"


# ─── Requests ─────────────────────────────────────────────────────

class GenerateRequest(BaseModel):
    """Request body for /api/v1/generate endpoint."""
    model_name: ModelName = Field(
        default=ModelName.ctgan,
        description="SDV model to use for generation",
    )
    num_rows: int = Field(
        default=10000,
        ge=100,
        le=1_000_000,
        description="Number of synthetic rows to generate",
    )
    apply_constraints: bool = Field(
        default=True,
        description="Apply African financial business constraints post-generation",
    )
    random_seed: int = Field(default=42, description="Random seed for reproducibility")


class AnalyzeRequest(BaseModel):
    """Request body for /api/v1/analyze endpoint."""
    detect_african_patterns: bool = Field(
        default=True,
        description="Run African-specific pattern detection (mobile money, seasonality)",
    )
    max_rows: Optional[int] = Field(
        default=None,
        description="Max rows to load from uploaded file (for large files)",
    )


# ─── Responses ────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    """Response for /api/v1/health."""
    status: str = "healthy"
    version: str
    timestamp: datetime


class JobStatus(str, Enum):
    pending = "pending"
    running = "running"
    completed = "completed"
    failed = "failed"


class JobResponse(BaseModel):
    """Response when a job is submitted."""
    job_id: str
    status: JobStatus
    message: str
    created_at: datetime


class ProfileResponse(BaseModel):
    """Response for dataset profiling."""
    name: str
    rows: int
    columns: int
    numeric_columns: int
    categorical_columns: int
    total_missing: int
    total_outliers: int
    african_patterns: Optional[dict] = None


class ValidationResponse(BaseModel):
    """Response for validation reports."""
    fidelity_score: float
    privacy_score: float
    combined_score: float
    verdict: str
    details: dict


class BenchmarkMetrics(BaseModel):
    """Metrics for a single model in a single scenario."""
    model: str
    roc_auc: float = 0.0
    f1: float = 0.0
    precision: float = 0.0
    recall: float = 0.0
    accuracy: float = 0.0


class BenchmarkScenario(BaseModel):
    """Results for one TSTR scenario."""
    scenario_name: str
    models: list[BenchmarkMetrics]


class BenchmarkResponse(BaseModel):
    """Full benchmark report."""
    scenarios: list[BenchmarkScenario]
    best_synthetic_auc: float = Field(
        description="Best ROC-AUC achieved by a model trained on synthetic data"
    )
    real_baseline_auc: float = Field(
        description="ROC-AUC of the best model trained on real data (baseline)"
    )
    utility_ratio: float = Field(
        description="Ratio of synthetic AUC / real AUC (closer to 1.0 = better)"
    )
