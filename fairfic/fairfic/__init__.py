from .core import FairnessInformationCriterion
from .metrics import compute_all_metrics, compute_group_metrics, compute_disparity
from .visualization import (
    set_publication_style,
    plot_fic_heatmap,
    plot_fic_heatmaps_grid,
    plot_benchmarking_tiers
)
from .evaluation import evaluate_model_fairness, create_default_model, analyze_dataset_fairness

__version__ = "0.1.0"
__author__ = "Dr. OJ. Akintande"
__all__ = [
    "FairnessInformationCriterion",
    "compute_all_metrics",
    "compute_group_metrics",
    "compute_disparity",
    "set_publication_style",
    "plot_fic_heatmap",
    "plot_fic_heatmaps_grid",
    "plot_benchmarking_tiers",
    "evaluate_model_fairness",
    "create_default_model", 
    "analyze_dataset_fairness"
]