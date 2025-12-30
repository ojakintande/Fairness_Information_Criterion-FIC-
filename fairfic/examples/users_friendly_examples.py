import numpy as np
import pandas as pd
from sklearn.datasets import make_classification

# Import FairFIC
from fairfic import (
    analyze_dataset_fairness,
    FairnessInformationCriterion,
    plot_fic_heatmaps_grid
)

print("="*70)
print("FAIRFIC: Fairness Analysis with ANY Dataset")
print("="*70)

# 1. User creates or loads their dataset
print("\n1. Creating example dataset...")
np.random.seed(42)

# Simulate user data: 1000 samples, 10 features
X, y = make_classification(
    n_samples=1000, n_features=10, n_informative=5,
    n_redundant=2, n_clusters_per_class=1, random_state=42
)

# Simulate protected attribute (e.g., 3 demographic groups)
# In real use, users would have their own groups like ['Male', 'Female', 'Other']
groups = np.random.choice(['Group_A', 'Group_B', 'Group_C'], size=1000, p=[0.5, 0.3, 0.2])

print(f" Samples: {X.shape[0]}, Features: {X.shape[1]}")
print(f" Target distribution: {pd.Series(y).value_counts().to_dict()}")
print(f" Group distribution: {pd.Series(groups).value_counts().to_dict()}")

# 2. User runs complete fairness analysis with ONE function call
print("\n2. Running fairness analysis...")
results = analyze_dataset_fairness(
    X=X,                    # User's features
    y=y,                    # User's labels
    groups=groups,          # User's protected attribute
    model_type='logistic',  # User can choose model type
    alphaF_values=[0.05, 0.10, 0.15]  # User can set fairness thresholds
)

print("✅ Analysis complete!")

# 3. User examines results
print("\n3. Analysis Results:")
print("-" * 40)

group_metrics = results['group_metrics']
print(f"\nModel Performance by Group:")
for group, metrics in group_metrics.items():
    print(f"  {group}: Accuracy={metrics['accuracy']:.3f}, "
          f"TPR={metrics['tpr']:.3f}, FPR={metrics['fpr']:.3f}")

# 4. User examines FIC results
fic_results = results['fic_results']['accuracy']  # For accuracy metric
fic = results['fic']

print(f"\nFairness Information Criterion (FIC) Analysis:")
for alphaF in [0.05, 0.10]:
    if alphaF in fic_results:
        pairs = list(fic_results[alphaF].keys())
        print(f"\n  For αF = {alphaF}:")
        for pair in pairs[:3]:  # Show first 3 pairs
            data = fic_results[alphaF][pair]
            print(f"    {pair}: FIC={data['fic_score']:.3f} ({data['tier']})")

# 5. User can create visualizations
print("\n4. Creating visualizations...")
try:
    fig, axes, cbar = plot_fic_heatmaps_grid(
        fic_results, 
        metric='accuracy',
        figsize=(15, 12)
    )
    
    # Save or show plot
    import matplotlib.pyplot as plt
    plt.savefig('fairness_analysis.png', dpi=300, bbox_inches='tight')
    print("✅ Saved plot: fairness_analysis.png")
    
except ImportError:
    print(" Failed: Matplotlib not installed, skipping plots")

print("\n" + "="*70)
print("ANALYSIS COMPLETE! 🎉")
print("="*70)

print("SUMMARY:")
print("-" * 40)
print("Users can use FairFIC with ANY dataset by:")
print("1. Providing features (X), labels (y), and groups")
print("2. Calling analyze_dataset_fairness()")
print("3. Examining results and creating visualizations")
print("\nNo need for specific dataset formats or pre-processing!")