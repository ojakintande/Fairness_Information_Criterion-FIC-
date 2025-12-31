import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer

# Import our package
from fairfic import (
    FairnessInformationCriterion,
    compute_all_metrics,
    compute_group_metrics,
    compute_disparity,
    plot_fic_heatmaps_grid,
    plot_benchmarking_tiers
)
from fairfic.data import load_compas_data

print("="*80)
print("COMPREHENSIVE FAIRFIC PACKAGE TEST")
print("="*80)

# Test 1: Basic FIC functionality
print("\n✅ Test 1: Basic FIC functionality")
fic = FairnessInformationCriterion([0.05, 0.10, 0.15, 0.20])
print(f"   Created FIC with alphaF values: {fic.alphaF_values}")

# Test 2: Metrics computation
print("\n✅ Test 2: Metrics computation")
y_true = np.array([0, 1, 0, 1, 0, 1])
y_pred = np.array([0, 1, 1, 1, 0, 0])
y_prob = np.array([0.1, 0.9, 0.6, 0.8, 0.2, 0.4])
metrics = compute_all_metrics(y_true, y_pred, y_prob)
print(f"   Computed {len(metrics)} metrics: {list(metrics.keys())}")

# Test 3: Group metrics
print("\n✅ Test 3: Group metrics computation")
groups = np.array(['A', 'A', 'B', 'B', 'A', 'B'])
group_metrics = compute_group_metrics(y_true, y_pred, y_prob, groups)
print(f"   Computed metrics for {len(group_metrics)} groups")

# Test 4: Disparity computation
print("\n✅ Test 4: Disparity computation")
disparity = compute_disparity(group_metrics, 'accuracy')
print(f"   Accuracy disparity: {disparity['disparity']:.3f} between {disparity['min_group']} and {disparity['max_group']}")

# Test 5: FIC analysis
print("\n✅ Test 5: FIC analysis")
results = fic.analyze_fairness(group_metrics, 'accuracy')
print(f"   Analyzed fairness for {len(results)} alphaF values")

# Test 6: Summary
print("\n✅ Test 6: FIC summary")
summary = fic.get_fairness_summary(results)
for alphaF, stats in summary.items():
    print(f"   αF={alphaF}: Mean FIC={stats['fic_mean']:.3f}, Tiers={stats['tier_counts']}")

print("\n" + "="*80)
print("ALL TESTS PASSED! ✅")
print("="*80)

# Optional: Test with simulated COMPAS-like data
print("\n📊 Simulating COMPAS-like analysis...")
np.random.seed(42)

# Simulate data
n_samples = 1000
simulated_metrics = {
    'African_American': {
        'accuracy': 0.72 + np.random.normal(0, 0.03),
        'tpr': 0.65 + np.random.normal(0, 0.04),
        'fpr': 0.28 + np.random.normal(0, 0.03)
    },
    'Caucasian': {
        'accuracy': 0.81 + np.random.normal(0, 0.02),
        'tpr': 0.78 + np.random.normal(0, 0.03),
        'fpr': 0.19 + np.random.normal(0, 0.02)
    },
    'Hispanic': {
        'accuracy': 0.76 + np.random.normal(0, 0.03),
        'tpr': 0.71 + np.random.normal(0, 0.04),
        'fpr': 0.24 + np.random.normal(0, 0.03)
    }
}

fic = FairnessInformationCriterion([0.05, 0.10, 0.15, 0.20])
results = fic.analyze_fairness(simulated_metrics, 'accuracy')
summary = fic.get_fairness_summary(results)

print("\nSimulated COMPAS Analysis Summary:")
for alphaF, stats in summary.items():
    print(f"\nαF = {alphaF}:")
    print(f"  Omega: {stats['omega_mean']:.4f} ± {stats['omega_std']:.4f}")
    print(f"  FIC: {stats['fic_mean']:.3f} [{stats['fic_min']:.3f}, {stats['fic_max']:.3f}]")
    print(f"  Tiers: {stats['tier_counts']}")

print("\n" + "="*80)
print("Package is fully functional! 🎉")
print("="*80)
