"""
Synthara — Full Pipeline Runner
Orchestrates the complete Synthara pipeline:
1. Load source data
2. Analyze patterns
3. Generate synthetic data
4. Validate fidelity + privacy
5. Run TSTR benchmark
6. Produce visualizations and reports

Usage:
    python scripts/run_pipeline.py [--source path] [--nrows N] [--model ctgan]
"""

import json
import logging
import sys
import time
from pathlib import Path

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

import pandas as pd

from src.data_analysis.african_patterns import AfricanPatternsDetector
from src.data_analysis.profiler import DataProfiler
from src.data_analysis.visualizer import DataVisualizer
from src.benchmark.tstr import TSTRBenchmark
from src.generation.synthesizer import SyntharaSynthesizer
from src.validation.reporter import ValidationReporter

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger("synthara.pipeline")


def run_pipeline(
    source_path: str,
    model_name: str = "ctgan",
    num_rows: int = 10000,
    nrows_load: int = None,
):
    """
    Run the complete Synthara pipeline.

    Args:
        source_path: Path to source CSV file.
        model_name: SDV model to use (ctgan, gaussian_copula, tvae).
        num_rows: Number of synthetic rows to generate.
        nrows_load: Max rows to load from source (for large files).
    """
    pipeline_start = time.time()

    logger.info("=" * 70)
    logger.info("🌍 SYNTHARA PIPELINE — Starting")
    logger.info("=" * 70)

    # ----------------------------------------------------------------
    # STEP 1: Load Data
    # ----------------------------------------------------------------
    logger.info("\n📥 STEP 1: Loading source data...")
    df = pd.read_csv(source_path, nrows=nrows_load)
    logger.info(f"Loaded {len(df)} rows × {df.shape[1]} columns from {source_path}")

    # ----------------------------------------------------------------
    # STEP 2: Analyze Patterns
    # ----------------------------------------------------------------
    logger.info("\n📊 STEP 2: Analyzing patterns...")

    # Statistical profiling
    profiler = DataProfiler()
    profile = profiler.profile(df, name="PaySim_Source")
    logger.info(f"Profile summary: {json.dumps(profile.summary(), indent=2)}")

    # African-specific patterns
    detector = AfricanPatternsDetector()
    patterns = detector.detect_patterns(df)
    logger.info(f"African patterns: {json.dumps(patterns.summary(), indent=2)}")

    # ----------------------------------------------------------------
    # STEP 3: Generate Synthetic Data
    # ----------------------------------------------------------------
    logger.info(f"\n🧠 STEP 3: Generating {num_rows} synthetic rows with {model_name}...")

    # Drop ID columns before generation (they don't carry useful patterns)
    gen_df = df.drop(columns=["nameOrig", "nameDest"], errors="ignore")

    synth = SyntharaSynthesizer(model_name=model_name, apply_constraints=True)
    synth.fit(gen_df)
    synthetic_df = synth.generate(num_rows=num_rows)

    # Save synthetic data
    output_dir = project_root / "data" / "generated"
    output_dir.mkdir(parents=True, exist_ok=True)
    synth_path = output_dir / f"synthetic_{model_name}.csv"
    synthetic_df.to_csv(synth_path, index=False)
    logger.info(f"Synthetic data saved to {synth_path}")

    # ----------------------------------------------------------------
    # STEP 4: Validate Fidelity + Privacy
    # ----------------------------------------------------------------
    logger.info("\n✅ STEP 4: Validating synthetic data...")

    reporter = ValidationReporter(output_dir=str(project_root / "data" / "reports"))
    validation_report = reporter.generate_report(
        gen_df, synthetic_df, model_name=model_name
    )

    logger.info(
        f"Validation: fidelity={validation_report['overall']['fidelity_score']:.4f}, "
        f"privacy={validation_report['overall']['privacy_score']:.4f}, "
        f"verdict={validation_report['overall']['verdict']}"
    )

    # ----------------------------------------------------------------
    # STEP 5: Visualize
    # ----------------------------------------------------------------
    logger.info("\n📈 STEP 5: Generating visualizations...")

    visualizer = DataVisualizer(output_dir=str(project_root / "data" / "reports"))

    # Distribution comparison
    numeric_cols = gen_df.select_dtypes(include=["number"]).columns.tolist()
    # Exclude step and flags, keep meaningful numeric columns
    plot_cols = [c for c in numeric_cols if c not in ["step", "isFlaggedFraud", "isFraud"]][:6]
    visualizer.plot_distributions(gen_df, synthetic_df, columns=plot_cols)

    # Correlation comparison
    visualizer.plot_correlation_comparison(gen_df, synthetic_df)

    # Categorical comparison (transaction types)
    if "type" in gen_df.columns and "type" in synthetic_df.columns:
        visualizer.plot_categorical_comparison(gen_df, synthetic_df, "type")

    # Fraud analysis
    if "isFraud" in gen_df.columns and "isFraud" in synthetic_df.columns:
        visualizer.plot_fraud_analysis(gen_df, synthetic_df)

    # ----------------------------------------------------------------
    # STEP 6: TSTR Benchmark
    # ----------------------------------------------------------------
    logger.info("\n🏆 STEP 6: Running TSTR benchmark...")

    # Re-add dummy ID columns for benchmark (it will drop them anyway)
    benchmark = TSTRBenchmark(
        target_column="isFraud",
        drop_columns=["isFlaggedFraud"],
        encode_columns=["type"],
    )
    benchmark_results = benchmark.run(gen_df, synthetic_df)

    # Save benchmark results
    benchmark_path = project_root / "data" / "reports" / "benchmark_results.json"
    with open(benchmark_path, "w", encoding="utf-8") as f:
        json.dump(benchmark_results, f, indent=2, default=str)

    # Visualize benchmark
    visualizer.plot_benchmark_results(benchmark_results)

    # ----------------------------------------------------------------
    # SUMMARY
    # ----------------------------------------------------------------
    total_time = time.time() - pipeline_start
    logger.info("\n" + "=" * 70)
    logger.info("🌍 SYNTHARA PIPELINE — Complete")
    logger.info("=" * 70)
    logger.info(f"  Total time: {total_time:.1f}s")
    logger.info(f"  Source: {len(df)} rows")
    logger.info(f"  Synthetic: {len(synthetic_df)} rows")
    logger.info(f"  Model: {model_name} (fit time: {synth.fit_time:.1f}s)")
    logger.info(f"  Fidelity: {validation_report['overall']['fidelity_score']:.4f}")
    logger.info(f"  Privacy: {validation_report['overall']['privacy_score']:.4f}")
    logger.info(f"  Verdict: {validation_report['overall']['verdict']}")
    logger.info(f"\n  📁 Outputs:")
    logger.info(f"     Synthetic data: {synth_path}")
    logger.info(f"     Reports: {project_root / 'data' / 'reports'}")
    logger.info("=" * 70)

    return {
        "synthetic_data_path": str(synth_path),
        "validation": validation_report["overall"],
        "benchmark": benchmark_results,
    }


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Synthara Pipeline Runner")
    parser.add_argument(
        "--source",
        default=str(project_root / "data" / "source" / "paysim_sample.csv"),
        help="Path to source CSV file",
    )
    parser.add_argument("--model", default="ctgan", choices=["ctgan", "gaussian_copula", "tvae"])
    parser.add_argument("--num-rows", type=int, default=10000, help="Number of synthetic rows")
    parser.add_argument("--nrows", type=int, default=None, help="Max source rows to load")

    args = parser.parse_args()

    run_pipeline(
        source_path=args.source,
        model_name=args.model,
        num_rows=args.num_rows,
        nrows_load=args.nrows,
    )
