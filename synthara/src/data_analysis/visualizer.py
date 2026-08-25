"""
Synthara — Data Visualizer
Generates publication-quality charts comparing real vs synthetic data:
- Distribution histograms (side-by-side)
- Correlation heatmaps
- Box plots by category
- Temporal patterns
- Fraud analysis visualizations
"""

import logging
from pathlib import Path
from typing import Optional

import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server use
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Synthara visual identity
SYNTHARA_COLORS = {
    "primary": "#6C63FF",    # Purple — brand color
    "secondary": "#00D2FF",  # Cyan — accent
    "real": "#FF6B6B",       # Coral — real data
    "synthetic": "#4ECDC4",  # Teal — synthetic data
    "background": "#1A1A2E", # Dark navy
    "text": "#E0E0E0",       # Light gray
    "grid": "#2D2D44",       # Subtle grid
}


def setup_synthara_style():
    """Apply Synthara's dark, premium visual style to matplotlib."""
    plt.style.use("dark_background")
    plt.rcParams.update({
        "figure.facecolor": SYNTHARA_COLORS["background"],
        "axes.facecolor": SYNTHARA_COLORS["background"],
        "axes.edgecolor": SYNTHARA_COLORS["grid"],
        "axes.labelcolor": SYNTHARA_COLORS["text"],
        "text.color": SYNTHARA_COLORS["text"],
        "xtick.color": SYNTHARA_COLORS["text"],
        "ytick.color": SYNTHARA_COLORS["text"],
        "grid.color": SYNTHARA_COLORS["grid"],
        "grid.alpha": 0.3,
        "font.size": 11,
        "axes.titlesize": 14,
        "figure.titlesize": 16,
    })


class DataVisualizer:
    """
    Generates comparative visualizations for real vs synthetic data.
    All charts use Synthara's premium dark theme.
    """

    def __init__(self, output_dir: str = "data/reports"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        setup_synthara_style()

    def plot_distributions(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        columns: Optional[list] = None,
        filename: str = "distributions_comparison.png",
    ) -> str:
        """
        Plot side-by-side distribution histograms for numeric columns.

        Args:
            real_df: Real/source DataFrame.
            synthetic_df: Synthetic DataFrame.
            columns: Specific columns to plot (defaults to all numeric).
            filename: Output filename.

        Returns:
            Path to saved figure.
        """
        if columns is None:
            columns = real_df.select_dtypes(include=[np.number]).columns.tolist()

        n_cols = min(len(columns), 6)  # Max 6 columns per figure
        columns = columns[:n_cols]

        n_rows = (n_cols + 1) // 2
        fig, axes = plt.subplots(n_rows, 2, figsize=(14, 4 * n_rows))
        fig.suptitle(
            "Distributions: Real vs Synthetic",
            fontsize=18,
            fontweight="bold",
            color=SYNTHARA_COLORS["primary"],
            y=1.02,
        )

        axes = axes.flatten() if n_cols > 2 else [axes] if n_cols == 1 else axes.flatten()

        for i, col in enumerate(columns):
            ax = axes[i]

            # Clip extreme outliers for better visualization
            real_data = real_df[col].dropna()
            synth_data = synthetic_df[col].dropna()

            p01, p99 = real_data.quantile(0.01), real_data.quantile(0.99)
            real_clipped = real_data.clip(p01, p99)
            synth_clipped = synth_data.clip(p01, p99)

            ax.hist(
                real_clipped, bins=50, alpha=0.6, density=True,
                color=SYNTHARA_COLORS["real"], label="Real", edgecolor="none",
            )
            ax.hist(
                synth_clipped, bins=50, alpha=0.6, density=True,
                color=SYNTHARA_COLORS["synthetic"], label="Synthetic", edgecolor="none",
            )
            ax.set_title(col, fontsize=12, color=SYNTHARA_COLORS["secondary"])
            ax.legend(fontsize=9)
            ax.grid(True, alpha=0.2)

        # Hide unused axes
        for j in range(len(columns), len(axes)):
            axes[j].set_visible(False)

        plt.tight_layout()
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Saved distribution comparison: {filepath}")
        return str(filepath)

    def plot_correlation_comparison(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        filename: str = "correlation_comparison.png",
    ) -> str:
        """
        Plot correlation heatmaps side by side for real vs synthetic data.
        """
        numeric_cols = real_df.select_dtypes(include=[np.number]).columns.tolist()
        common_cols = [c for c in numeric_cols if c in synthetic_df.columns][:8]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 7))
        fig.suptitle(
            "Correlation Matrix: Real vs Synthetic",
            fontsize=18,
            fontweight="bold",
            color=SYNTHARA_COLORS["primary"],
        )

        # Real correlations
        real_corr = real_df[common_cols].corr()
        sns.heatmap(
            real_corr, ax=ax1, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, square=True,
            annot_kws={"size": 8},
        )
        ax1.set_title("Real Data", color=SYNTHARA_COLORS["real"], fontsize=14)

        # Synthetic correlations
        synth_corr = synthetic_df[common_cols].corr()
        sns.heatmap(
            synth_corr, ax=ax2, annot=True, fmt=".2f", cmap="coolwarm",
            center=0, vmin=-1, vmax=1, square=True,
            annot_kws={"size": 8},
        )
        ax2.set_title("Synthetic Data", color=SYNTHARA_COLORS["synthetic"], fontsize=14)

        plt.tight_layout()
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Saved correlation comparison: {filepath}")
        return str(filepath)

    def plot_categorical_comparison(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        column: str,
        filename: Optional[str] = None,
    ) -> str:
        """
        Plot bar chart comparing categorical distributions.
        """
        if filename is None:
            filename = f"categorical_{column}.png"

        real_counts = real_df[column].value_counts(normalize=True).sort_index()
        synth_counts = synthetic_df[column].value_counts(normalize=True).sort_index()

        # Align categories
        all_cats = sorted(set(real_counts.index) | set(synth_counts.index))
        real_vals = [real_counts.get(c, 0) for c in all_cats]
        synth_vals = [synth_counts.get(c, 0) for c in all_cats]

        x = np.arange(len(all_cats))
        width = 0.35

        fig, ax = plt.subplots(figsize=(12, 6))
        bars1 = ax.bar(
            x - width / 2, real_vals, width,
            label="Real", color=SYNTHARA_COLORS["real"], alpha=0.8,
        )
        bars2 = ax.bar(
            x + width / 2, synth_vals, width,
            label="Synthetic", color=SYNTHARA_COLORS["synthetic"], alpha=0.8,
        )

        ax.set_xlabel(column, fontsize=12)
        ax.set_ylabel("Proportion", fontsize=12)
        ax.set_title(
            f"Distribution: {column} (Real vs Synthetic)",
            fontsize=14,
            color=SYNTHARA_COLORS["primary"],
            fontweight="bold",
        )
        ax.set_xticks(x)
        ax.set_xticklabels(all_cats, rotation=45, ha="right")
        ax.legend()
        ax.grid(True, axis="y", alpha=0.2)

        plt.tight_layout()
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Saved categorical comparison: {filepath}")
        return str(filepath)

    def plot_fraud_analysis(
        self,
        real_df: pd.DataFrame,
        synthetic_df: pd.DataFrame,
        fraud_col: str = "isFraud",
        type_col: str = "type",
        amount_col: str = "amount",
        filename: str = "fraud_analysis.png",
    ) -> str:
        """
        Generate a 2x2 fraud analysis dashboard.
        """
        fig, axes = plt.subplots(2, 2, figsize=(16, 12))
        fig.suptitle(
            "Fraud Pattern Analysis: Real vs Synthetic",
            fontsize=18,
            fontweight="bold",
            color=SYNTHARA_COLORS["primary"],
        )

        # 1. Fraud rate comparison
        ax = axes[0, 0]
        real_fraud_rate = real_df[fraud_col].mean() * 100
        synth_fraud_rate = synthetic_df[fraud_col].mean() * 100
        bars = ax.bar(
            ["Real", "Synthetic"],
            [real_fraud_rate, synth_fraud_rate],
            color=[SYNTHARA_COLORS["real"], SYNTHARA_COLORS["synthetic"]],
            alpha=0.8,
        )
        ax.set_ylabel("Fraud Rate (%)")
        ax.set_title("Overall Fraud Rate", color=SYNTHARA_COLORS["secondary"])
        for bar, val in zip(bars, [real_fraud_rate, synth_fraud_rate]):
            ax.text(
                bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.001,
                f"{val:.4f}%", ha="center", va="bottom", fontsize=10,
            )

        # 2. Fraud by transaction type
        ax = axes[0, 1]
        if type_col in real_df.columns:
            real_fraud_by_type = real_df.groupby(type_col)[fraud_col].mean() * 100
            synth_fraud_by_type = synthetic_df.groupby(type_col)[fraud_col].mean() * 100
            types = sorted(set(real_fraud_by_type.index) | set(synth_fraud_by_type.index))
            x = np.arange(len(types))
            width = 0.35
            ax.bar(
                x - width / 2,
                [real_fraud_by_type.get(t, 0) for t in types],
                width, label="Real", color=SYNTHARA_COLORS["real"], alpha=0.8,
            )
            ax.bar(
                x + width / 2,
                [synth_fraud_by_type.get(t, 0) for t in types],
                width, label="Synthetic", color=SYNTHARA_COLORS["synthetic"], alpha=0.8,
            )
            ax.set_xticks(x)
            ax.set_xticklabels(types, rotation=45, ha="right")
            ax.legend()
        ax.set_ylabel("Fraud Rate (%)")
        ax.set_title("Fraud Rate by Type", color=SYNTHARA_COLORS["secondary"])

        # 3. Fraud vs Legit amount distribution (real)
        ax = axes[1, 0]
        if amount_col in real_df.columns:
            fraud_amounts = real_df[real_df[fraud_col] == 1][amount_col]
            legit_amounts = real_df[real_df[fraud_col] == 0][amount_col]
            if len(fraud_amounts) > 0:
                p99 = fraud_amounts.quantile(0.99)
                ax.hist(
                    fraud_amounts.clip(0, p99), bins=50, alpha=0.6,
                    color="#FF4444", label="Fraud", density=True, edgecolor="none",
                )
            if len(legit_amounts) > 0:
                p99 = legit_amounts.quantile(0.99)
                ax.hist(
                    legit_amounts.clip(0, p99), bins=50, alpha=0.4,
                    color="#44FF44", label="Legit", density=True, edgecolor="none",
                )
            ax.legend()
        ax.set_xlabel("Amount")
        ax.set_title("Amount Distribution (Real)", color=SYNTHARA_COLORS["secondary"])

        # 4. Fraud vs Legit amount distribution (synthetic)
        ax = axes[1, 1]
        if amount_col in synthetic_df.columns:
            fraud_amounts = synthetic_df[synthetic_df[fraud_col] == 1][amount_col]
            legit_amounts = synthetic_df[synthetic_df[fraud_col] == 0][amount_col]
            if len(fraud_amounts) > 0:
                p99 = fraud_amounts.quantile(0.99)
                ax.hist(
                    fraud_amounts.clip(0, p99), bins=50, alpha=0.6,
                    color="#FF4444", label="Fraud", density=True, edgecolor="none",
                )
            if len(legit_amounts) > 0:
                p99 = legit_amounts.quantile(0.99)
                ax.hist(
                    legit_amounts.clip(0, p99), bins=50, alpha=0.4,
                    color="#44FF44", label="Legit", density=True, edgecolor="none",
                )
            ax.legend()
        ax.set_xlabel("Amount")
        ax.set_title("Amount Distribution (Synthetic)", color=SYNTHARA_COLORS["secondary"])

        plt.tight_layout()
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Saved fraud analysis: {filepath}")
        return str(filepath)

    def plot_benchmark_results(
        self,
        results: dict,
        filename: str = "benchmark_results.png",
    ) -> str:
        """
        Plot benchmark comparison chart (TSTR results).

        Args:
            results: Dict with structure:
                {
                    "scenario_name": {
                        "model_name": {"roc_auc": 0.95, "f1": 0.87, ...}
                    }
                }
        """
        scenarios = list(results.keys())
        if not scenarios:
            logger.warning("No benchmark results to plot")
            return ""

        # Get all models and metrics
        all_models = set()
        for scenario in scenarios:
            all_models.update(results[scenario].keys())
        all_models = sorted(all_models)

        metrics = ["roc_auc", "f1", "precision", "recall"]

        fig, axes = plt.subplots(1, len(metrics), figsize=(5 * len(metrics), 6))
        fig.suptitle(
            "Benchmark: Synthetic vs Real Training",
            fontsize=18,
            fontweight="bold",
            color=SYNTHARA_COLORS["primary"],
        )

        scenario_colors = [
            SYNTHARA_COLORS["real"],
            SYNTHARA_COLORS["synthetic"],
            SYNTHARA_COLORS["secondary"],
        ]

        for idx, metric in enumerate(metrics):
            ax = axes[idx] if len(metrics) > 1 else axes
            x = np.arange(len(all_models))
            width = 0.25

            for s_idx, scenario in enumerate(scenarios):
                values = [
                    results[scenario].get(model, {}).get(metric, 0)
                    for model in all_models
                ]
                color = scenario_colors[s_idx % len(scenario_colors)]
                ax.bar(
                    x + s_idx * width - width, values, width,
                    label=scenario, color=color, alpha=0.8,
                )

            ax.set_ylabel(metric.upper())
            ax.set_title(metric.replace("_", " ").upper(), color=SYNTHARA_COLORS["secondary"])
            ax.set_xticks(x)
            ax.set_xticklabels(all_models, rotation=45, ha="right", fontsize=9)
            ax.set_ylim(0, 1.05)
            ax.grid(True, axis="y", alpha=0.2)
            if idx == 0:
                ax.legend(fontsize=8)

        plt.tight_layout()
        filepath = self.output_dir / filename
        fig.savefig(filepath, dpi=150, bbox_inches="tight", facecolor=fig.get_facecolor())
        plt.close(fig)
        logger.info(f"Saved benchmark results: {filepath}")
        return str(filepath)
