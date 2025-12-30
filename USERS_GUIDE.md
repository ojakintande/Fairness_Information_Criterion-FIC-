# Create comprehensive user guide
@'
# FairFIC User Guide
## Fairness Information Criterion for Algorithmic Fairness Evaluation

**Version:** 0.1.0  
**Author:** Dr. OJ Akintande  
**Date:** $(Get-Date -Format "yyyy-MM-dd")  
**Paper Status:** Under Review (IMCL2026)

---

## Table of Contents
1. [Introduction](#introduction)
2. [Installation](#installation)
3. [Quick Start Examples](#quick-start-examples)
4. [Core Concepts](#core-concepts)
5. [API Reference](#api-reference)
6. [Advanced Usage](#advanced-usage)
7. [Case Study: COMPAS Dataset](#case-study-compas-dataset)
8. [Troubleshooting](#troubleshooting)
9. [Citation](#citation)

---

## Introduction

FairFIC (Fairness Information Criterion) is a Python package for evaluating algorithmic fairness using the Fairness Information Criterion (FIC). The FIC quantifies fairness between demographic groups and classifies fairness into tiers: Optimum, Acceptable, Questionable, and Unacceptable.

### Key Features
- **FIC Algorithm**: Quantifies fairness with configurable thresholds (αF)
- **Comprehensive Metrics**: Accuracy, TPR, FPR, F1, AUC, and more
- **Visualization**: Publication-ready heatmaps and benchmarking plots
- **User-Friendly API**: One-line analysis with `analyze_dataset_fairness()`
- **Dataset Support**: Built-in COMPAS dataset loader
- **Model Agnostic**: Works with any scikit-learn compatible model

---

## Installation

### Prerequisites
- Python 3.8 or higher
- pip package manager

### Basic Installation
```bash
# Install from source
git clone https://github.com/yourusername/fairfic.git
cd fairfic
pip install -e .

#  Example 1: One-Line Complete Analysis

import numpy as np
from fairfic import analyze_dataset_fairness

# Generate synthetic data
np.random.seed(42)
X = np.random.randn(1000, 10)  # 1000 samples, 10 features
y = np.random.choice([0, 1], 1000)  # Binary target
groups = np.random.choice(['Group_A', 'Group_B', 'Group_C'], 
                         1000, p=[0.4, 0.4, 0.2])  # Protected attribute

# Complete fairness analysis in one line
results = analyze_dataset_fairness(
    X=X,                    # Features
    y=y,                    # Labels  
    groups=groups,          # Protected attribute
    model_type='logistic',  # Model type
    alphaF_values=[0.05, 0.10, 0.15]  # Fairness thresholds
)

# Examine results
print("Model Performance by Group:")
for group, metrics in results['group_metrics'].items():
    print(f"  {group}: Accuracy={metrics['accuracy']:.3f}, "
          f"TPR={metrics['tpr']:.3f}")

print("\nFairness Analysis (Accuracy Metric):")
fic_results = results['fic_results']['accuracy']
for alphaF in [0.05, 0.10]:
    if alphaF in fic_results:
        print(f"\n  For αF = {alphaF}:")
        for pair, data in fic_results[alphaF].items():
            print(f"    {pair}: FIC={data['fic_score']:.3f} ({data['tier']})")

#... Example 2: Step-by-Step Analysis with Custom Model

import numpy as np
from sklearn.ensemble import RandomForestClassifier
from fairfic import (
    evaluate_model_fairness,
    FairnessInformationCriterion,
    plot_fic_heatmaps_grid
)

# 1. Prepare your data
X_train, X_test, y_train, y_test = ...  # Your train/test split
groups_train, groups_test = ...         # Your protected attribute

# 2. Train your custom model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# 3. Evaluate fairness
group_metrics, test_data = evaluate_model_fairness(
    model=model,
    X=X_test,
    y=y_test,
    groups=groups_test
)

# 4. Apply FIC
fic = FairnessInformationCriterion(alphaF_values=[0.05, 0.10, 0.15, 0.20])
results = fic.analyze_fairness(group_metrics, metric_name='accuracy')

# 5. Get summary statistics
summary = fic.get_fairness_summary(results)
for alphaF, stats in summary.items():
    print(f"αF={alphaF}: Mean FIC={stats['fic_mean']:.3f}, "
          f"Tiers: {stats['tier_counts']}")

# 6. Visualize
fig, axes, cbar = plot_fic_heatmaps_grid(results, metric='accuracy')
plt.savefig('fairness_analysis.png', dpi=300, bbox_inches='tight')

#... Example 3: Using Pre-computed Group Metrics

from fairfic import FairnessInformationCriterion

# If you already have performance metrics by group
group_metrics = {
    'African_American': {'accuracy': 0.75, 'tpr': 0.70, 'fpr': 0.25},
    'Caucasian': {'accuracy': 0.80, 'tpr': 0.75, 'fpr': 0.20},
    'Hispanic': {'accuracy': 0.78, 'tpr': 0.73, 'fpr': 0.22}
}

# Initialize FIC with fairness thresholds
fic = FairnessInformationCriterion(alphaF_values=[0.05, 0.10, 0.15, 0.20])

# Analyze for different metrics
for metric in ['accuracy', 'tpr', 'fpr']:
    print(f"\nAnalyzing {metric.upper()}:")
    results = fic.analyze_fairness(group_metrics, metric_name=metric)
    
    for alphaF in [0.05, 0.10]:
        if alphaF in results:
            print(f"\n  αF = {alphaF}:")
            for pair, data in results[alphaF].items():
                g1, g2 = pair.split(' - ')
                print(f"    {pair}: ω={data['omega']:.3f}, "
                      f"FIC={data['fic_score']:.3f} ({data['tier']})")


#... Core Concepts

Refer to the FIC Paper

# API Reference

## Main Functions

analyze_dataset_fairness(X, y, groups, **kwargs)
Complete fairness analysis pipeline.

Parameters:

X: Feature matrix (n_samples × n_features)

y: Target labels (binary)

groups: Protected attribute values

model_type: 'logistic', 'random_forest', 'l1', or 'l2'

alphaF_values: List of αF thresholds (default: [0.05, 0.10, 0.15, 0.20])

Returns: Dictionary with results

evaluate_model_fairness(model, X, y, groups, **kwargs)
Evaluate fairness of a trained model.

compute_group_metrics(y_true, y_pred, y_prob, groups)
Compute performance metrics for each group.

compute_disparity(group_metrics, metric_name)
Compute maximum disparity across groups.

Core Class
FairnessInformationCriterion(alphaF_values=None)
Main FIC implementation.

Methods:

analyze_fairness(group_metrics, metric_name): Analyze fairness

get_fairness_summary(fic_results): Generate statistical summary

compute_omega(metric1, metric2): Compute absolute difference

compute_fic(omega, alphaF): Compute FIC score

classify_tier(fic_score): Classify into fairness tier

Visualization Functions
plot_fic_heatmaps_grid(fic_results, metric, figsize)
Create grid of FIC heatmaps for all αF values.

plot_benchmarking_tiers(fic_results, alphaF, metric, figsize)
Create benchmarking tier plot for specific αF.

set_publication_style()
Set publication-quality plotting style.

#  Advanced Usage

#... Custom Metric Integration

from fairfic import FairnessInformationCriterion

# Define custom metric computation
def compute_custom_metric(y_true, y_pred):
    # Your custom metric implementation
    return custom_score

# Compute custom metric for each group
group_custom_metrics = {}
for group in unique_groups:
    mask = groups == group
    group_custom_metrics[group] = {
        'custom_metric': compute_custom_metric(y_true[mask], y_pred[mask])
    }

# Analyze with FIC
fic = FairnessInformationCriterion([0.05, 0.10])
results = fic.analyze_fairness(group_custom_metrics, 'custom_metric')

#..  Batch Analysis for Multiple Models

from fairfic import analyze_dataset_fairness
import pandas as pd

models = ['logistic', 'random_forest', 'l1', 'l2']
results_list = []

for model_type in models:
    print(f"\nAnalyzing {model_type}...")
    results = analyze_dataset_fairness(
        X, y, groups, 
        model_type=model_type,
        alphaF_values=[0.10]
    )
    
    # Extract summary
    fic_summary = results['fic'].get_fairness_summary(
        results['fic_results']['accuracy']
    )
    
    results_list.append({
        'Model': model_type,
        'Accuracy': results['group_metrics']['Overall']['accuracy'],
        'Mean_FIC': fic_summary[0.10]['fic_mean'],
        'Fairness_Tier': 'Optimum' if fic_summary[0.10]['fic_mean'] > 0.75 else 'Acceptable'
    })

# Compare models
comparison_df = pd.DataFrame(results_list)
print(comparison_df.to_string(index=False))

# Data Loader example

from fairfic.data import load_compas_data

# Load dataset
df = load_compas_data('compas-scores-two-years.csv')

# Check dataset
print(f"Dataset shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nTarget distribution:")
print(df['high_risk'].value_counts(normalize=True))
print(f"\nRace group distribution:")
print(df['race_group'].value_counts(normalize=True))


from fairfic import analyze_dataset_fairness
from fairfic.data import load_compas_data

# Load and prepare data
df = load_compas_data('compas-scores-two-years.csv')
X = df.drop(columns=['high_risk', 'race_group'])
y = df['high_risk']
groups = df['race_group']

# Run analysis
results = analyze_dataset_fairness(
    X=X, y=y, groups=groups,
    model_type='logistic',
    alphaF_values=[0.05, 0.10, 0.15, 0.20]
)

# Generate report
print("="*60)
print("COMPAS DATASET FAIRNESS REPORT")
print("="*60)

print("\n1. MODEL PERFORMANCE BY RACE GROUP")
print("-" * 40)
group_metrics = results['group_metrics']
for group, metrics in sorted(group_metrics.items()):
    print(f"{group:<20} Accuracy: {metrics['accuracy']:.3f} | "
          f"TPR: {metrics['tpr']:.3f} | FPR: {metrics['fpr']:.3f}")

print("\n2. FAIRNESS ANALYSIS (Accuracy Metric)")
print("-" * 40)
fic_results = results['fic_results']['accuracy']

for alphaF in [0.05, 0.10, 0.15]:
    if alphaF in fic_results:
        unfair_pairs = []
        for pair, data in fic_results[alphaF].items():
            if data['tier'] in ['Questionable', 'Unacceptable']:
                unfair_pairs.append((pair, data['fic_score'], data['tier']))
        
        if unfair_pairs:
            print(f"\nFor αF = {alphaF} (Strictness: {'High' if alphaF==0.05 else 'Medium' if alphaF==0.10 else 'Low'}):")
            for pair, fic_score, tier in unfair_pairs:
                print(f"  ⚠️  {pair}: FIC={fic_score:.3f} ({tier})")
        else:
            print(f"\nFor αF = {alphaF}: All pairs show acceptable fairness")

print("\n3. RECOMMENDATIONS")
print("-" * 40)
print("Based on FIC analysis:")
print("• Consider model retraining with fairness constraints")
print("• Monitor groups with Questionable/Unacceptable FIC scores")
print("• For high-stakes decisions, use αF = 0.05 threshold")
print("• Regular fairness audits recommended")

#  Visualizing COMPAS Results

from fairfic import plot_fic_heatmaps_grid, plot_benchmarking_tiers
import matplotlib.pyplot as plt

# Heatmaps for all alphaF values
fig1, axes1, cbar1 = plot_fic_heatmaps_grid(
    results['fic_results']['accuracy'],
    metric='accuracy',
    figsize=(20, 16)
)
plt.savefig('compas_fic_heatmaps.png', dpi=300, bbox_inches='tight')

# Benchmarking tiers for αF=0.10
fig2 = plot_benchmarking_tiers(
    results['fic_results']['accuracy'],
    alphaF=0.10,
    metric='accuracy',
    figsize=(16, 8)
)
plt.savefig('compas_benchmarking_tiers.png', dpi=300, bbox_inches='tight')


# Troubleshooting
# Common Issues and Solutions
# Issue 1: Import Errors

ModuleNotFoundError: No module named 'fairfic'

# Solution:

# Ensure package is installed
pip install -e .

# Check Python path
python -c "import sys; print(sys.path)"

#  Issue 2: Missing Dependencies

ImportError: No module named 'sklearn'

#  Solution:

pip install scikit-learn numpy pandas matplotlib seaborn

#  Issue 3: Data Format Issues

ValueError: Found array with dim 3. Expected <= 2

#... Solution: Ensure your data is 2D (samples × features):

X = np.array(X).reshape(-1, n_features)  # Reshape if needed

#... Issue 4: Protected Attribute Issues

KeyError: 'Group not found in metrics'

#  Solution: Check group labels match:

print(f"Unique groups: {np.unique(groups)}")
print(f"Groups in metrics: {list(group_metrics.keys())}")

#  Getting Help

Check examples in examples/ directory

Review API documentation above

Ensure data is properly formatted

Verify all dependencies are installed

#  Contact

Author: Dr. OJ Akintande

Email: ojoak@dtu.dk | aojsoft@gmail.com

Repository: https://github.com/ojakintande/fairfic



## **📄 STEP 2: Create PDF Conversion Script**

Now let's create a script to convert the markdown to PDF:

```powershell
# Create PDF conversion script
@'
#!/usr/bin/env python3
"""
Convert USER_GUIDE.md to PDF format.
Requires: pip install markdown pdfkit wkhtmltopdf
"""

import os
import sys
import markdown
import pdfkit
from datetime import datetime

def convert_markdown_to_pdf(md_file, pdf_file):
    """Convert markdown file to PDF."""
    
    # Read markdown content
    with open(md_file, 'r', encoding='utf-8') as f:
        md_content = f.read()
    
    # Convert markdown to HTML
    html_content = markdown.markdown(md_content, extensions=['extra', 'tables'])
    
    # Create full HTML document
    full_html = f'''
    <!DOCTYPE html>
    <html>
    <head>
        <meta charset="UTF-8">
        <title>FairFIC User Guide</title>
        <style>
            body {{
                font-family: Arial, sans-serif;
                line-height: 1.6;
                margin: 40px;
                max-width: 1000px;
                margin: 0 auto;
                padding: 40px;
            }}
            h1 {{
                color: #2c3e50;
                border-bottom: 3px solid #3498db;
                padding-bottom: 10px;
            }}
            h2 {{
                color: #34495e;
                border-bottom: 1px solid #bdc3c7;
                padding-bottom: 5px;
                margin-top: 30px;
            }}
            h3 {{
                color: #2c3e50;
            }}
            code {{
                background-color: #f8f9fa;
                padding: 2px 6px;
                border-radius: 3px;
                font-family: "Courier New", monospace;
            }}
            pre {{
                background-color: #f8f9fa;
                padding: 15px;
                border-radius: 5px;
                overflow-x: auto;
                border-left: 4px solid #3498db;
            }}
            table {{
                border-collapse: collapse;
                width: 100%;
                margin: 20px 0;
            }}
            th, td {{
                border: 1px solid #ddd;
                padding: 12px;
                text-align: left;
            }}
            th {{
                background-color: #3498db;
                color: white;
            }}
            tr:nth-child(even) {{
                background-color: #f2f2f2;
            }}
            .footer {{
                margin-top: 50px;
                padding-top: 20px;
                border-top: 1px solid #ddd;
                font-size: 0.9em;
                color: #7f8c8d;
                text-align: center;
            }}
        </style>
    </head>
    <body>
        <h1>FairFIC User Guide</h1>
        <p><strong>Fairness Information Criterion for Algorithmic Fairness Evaluation</strong></p>
        <p><strong>Version:</strong> 0.1.0<br>
        <strong>Author:</strong> Dr. OJ Akintande<br>
        <strong>Date:</strong> {datetime.now().strftime("%Y-%m-%d")}<br>
        <strong>Paper Status:</strong> Under Review</p>
        
        {html_content}
        
        <div class="footer">
            <p>Generated from USER_GUIDE.md on {datetime.now().strftime("%Y-%m-%d at %H:%M:%S")}</p>
            <p>© 2025 Dr. OJ Akintande. All rights reserved.</p>
        </div>
    </body>
    </html>
    '''
    
    # Convert HTML to PDF
    try:
        # Try using wkhtmltopdf
        options = {
            'page-size': 'A4',
            'margin-top': '0.75in',
            'margin-right': '0.75in',
            'margin-bottom': '0.75in',
            'margin-left': '0.75in',
            'encoding': "UTF-8",
            'no-outline': None,
            'enable-local-file-access': None,
        }
        
        pdfkit.from_string(full_html, pdf_file, options=options)
        print(f"✅ PDF created: {pdf_file}")
        
    except Exception as e:
        print(f"❌ PDF conversion failed: {e}")
        print("\nAlternative: Save as HTML and print to PDF manually")
        
        # Save as HTML instead
        html_file = pdf_file.replace('.pdf', '.html')
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(full_html)
        print(f"✅ HTML saved: {html_file}")
        print("You can open this in a browser and print to PDF")

def main():
    """Main function."""
    md_file = "USER_GUIDE.md"
    pdf_file = "FairFIC_User_Guide.pdf"
    
    if not os.path.exists(md_file):
        print(f"❌ Error: {md_file} not found")
        sys.exit(1)
    
    print("Converting USER_GUIDE.md to PDF...")
    convert_markdown_to_pdf(md_file, pdf_file)

if __name__ == "__main__":
    main()
'@ | Out-File -FilePath "create_pdf_guide.py" -Encoding utf8

Write-Host "✅ Created PDF conversion script"


 #   STEP 3: Create Installation Script for PDF Dependencies



