"""
Synthara — API Service Layer
Business logic invoked by FastAPI routes. Manages state, file I/O,
and orchestration of the data pipeline modules.
"""

import logging
import shutil
import uuid
from datetime import datetime
from pathlib import Path
from typing import Optional

import pandas as pd

from ..data_analysis.profiler import DataProfiler
from ..data_analysis.african_patterns import AfricanPatternsDetector
from ..generation.synthesizer import SyntharaSynthesizer
from ..validation.reporter import ValidationReporter

logger = logging.getLogger(__name__)

# Directories
PROJECT_ROOT = Path(__file__).parent.parent.parent
DATA_SOURCE = PROJECT_ROOT / "data" / "source"
DATA_GENERATED = PROJECT_ROOT / "data" / "generated"
DATA_REPORTS = PROJECT_ROOT / "data" / "reports"

for d in [DATA_SOURCE, DATA_GENERATED, DATA_REPORTS]:
    d.mkdir(parents=True, exist_ok=True)

# In-memory job store (production would use Redis/DB)
_jobs: dict = {}


class SyntharaService:
    """
    Orchestrates Synthara operations for the API layer.
    """

    def __init__(self):
        self.profiler = DataProfiler()
        self.pattern_detector = AfricanPatternsDetector()
        self.reporter = ValidationReporter(output_dir=str(DATA_REPORTS))

    def analyze_dataset(
        self,
        df: pd.DataFrame,
        name: str = "uploaded",
        detect_african: bool = True,
    ) -> dict:
        """
        Profile a dataset and optionally detect African-specific patterns.
        """
        profile = self.profiler.profile(df, name=name)
        result = profile.summary()

        if detect_african:
            patterns = self.pattern_detector.detect_patterns(df)
            result["african_patterns"] = patterns.summary()

        return result

    def generate_synthetic(
        self,
        source_df: pd.DataFrame,
        model_name: str = "ctgan",
        num_rows: int = 10000,
        apply_constraints: bool = True,
        random_seed: int = 42,
    ) -> tuple[pd.DataFrame, dict]:
        """
        Generate synthetic data from a source DataFrame.

        Returns:
            Tuple of (synthetic_df, validation_report).
        """
        # Drop ID columns before generation
        gen_df = source_df.drop(columns=["nameOrig", "nameDest"], errors="ignore")

        # Generate
        synth = SyntharaSynthesizer(
            model_name=model_name,
            apply_constraints=apply_constraints,
            random_seed=random_seed,
        )
        synth.fit(gen_df)
        synthetic_df = synth.generate(num_rows=num_rows)

        # Validate
        report = self.reporter.generate_report(
            gen_df, synthetic_df, model_name=model_name, save=True
        )

        return synthetic_df, report

    def save_uploaded_file(self, content: bytes, filename: str) -> Path:
        """Save an uploaded file to the source data directory."""
        filepath = DATA_SOURCE / filename
        filepath.write_bytes(content)
        logger.info(f"Saved uploaded file: {filepath}")
        return filepath

    def save_synthetic_csv(self, df: pd.DataFrame, model_name: str) -> Path:
        """Save generated synthetic data to CSV."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"synthetic_{model_name}_{timestamp}.csv"
        filepath = DATA_GENERATED / filename
        df.to_csv(filepath, index=False)
        logger.info(f"Saved synthetic data: {filepath} ({len(df)} rows)")
        return filepath

    def create_job(self, job_type: str) -> str:
        """Create a new job entry and return its ID."""
        job_id = str(uuid.uuid4())[:8]
        _jobs[job_id] = {
            "id": job_id,
            "type": job_type,
            "status": "pending",
            "created_at": datetime.now().isoformat(),
            "result": None,
            "error": None,
        }
        return job_id

    def update_job(self, job_id: str, status: str, result: dict = None, error: str = None):
        """Update a job's status."""
        if job_id in _jobs:
            _jobs[job_id]["status"] = status
            _jobs[job_id]["result"] = result
            _jobs[job_id]["error"] = error

    def get_job(self, job_id: str) -> Optional[dict]:
        """Get a job by ID."""
        return _jobs.get(job_id)
