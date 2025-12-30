from fairfic import FairnessInformationCriterion
import numpy as np

def main():
    print("=== Fairness Information Criterion (FIC) Demo ===\n")
    
    # Initialize FIC with different alphaF values
    fic = FairnessInformationCriterion(alphaF_values=[0.05, 0.10, 0.15, 0.20])
    
    # Simulate model performance metrics for different demographic groups
    np.random.seed(42)
    
    group_metrics = {
        'African_American': {
            'accuracy': 0.75 + np.random.normal(0, 0.02),
            'tpr': 0.70 + np.random.normal(0, 0.03),
            'fpr': 0.25 + np.random.normal(0, 0.02)
        },
        'Caucasian': {
            'accuracy': 0.80 + np.random.normal(0, 0.02),
            'tpr': 0.75 + np.random.normal(0, 0.03),
            'fpr': 0.20 + np.random.normal(0, 0.02)
        },
        'Hispanic': {
            'accuracy': 0.78 + np.random.normal(0, 0.02),
            'tpr': 0.73 + np.random.normal(0, 0.03),
            'fpr': 0.22 + np.random.normal(0, 0.02)
        }
    }
    
    print("Simulated Group Metrics:")
    print("-" * 50)
    for group, metrics in group_metrics.items():
        print(f"\n{group}:")
        for metric, value in metrics.items():
            print(f"  {metric.upper()}: {value:.3f}")
    
    print("\n" + "="*60)
    print("Fairness Analysis for Different Metrics")
    print("="*60)
    
    # Analyze fairness for different metrics
    for metric in ['accuracy', 'tpr', 'fpr']:
        print(f"\n📊 Metric: {metric.upper()}")
        print("-" * 40)
        
        results = fic.analyze_fairness(group_metrics, metric_name=metric)
        
        for alphaF, pairs_data in results.items():
            print(f"\n  αF = {alphaF}:")
            for pair, data in pairs_data.items():
                g1, g2 = pair.split(' - ')
                print(f"    {pair}:")
                print(f"      ω = {data['omega']:.4f}")
                print(f"      FIC = {data['fic_score']:.3f} ({data['tier']})")
                print(f"      Values: {g1}={data[f'{g1}_value']:.3f}, {g2}={data[f'{g2}_value']:.3f}")
    
    # Get summary for accuracy
    print("\n" + "="*60)
    print("Summary Statistics (Accuracy Metric)")
    print("="*60)
    
    results = fic.analyze_fairness(group_metrics, 'accuracy')
    summary = fic.get_fairness_summary(results)
    
    for alphaF, stats in summary.items():
        print(f"\nαF = {alphaF}:")
        print(f"  Omega Statistics:")
        print(f"    Mean: {stats['omega_mean']:.4f}")
        print(f"    Range: [{stats['omega_min']:.4f}, {stats['omega_max']:.4f}]")
        print(f"    Std: {stats['omega_std']:.4f}")
        print(f"  FIC Statistics:")
        print(f"    Mean: {stats['fic_mean']:.3f}")
        print(f"    Range: [{stats['fic_min']:.3f}, {stats['fic_max']:.3f}]")
        print(f"  Tier Distribution: {stats['tier_counts']}")
        print(f"  Number of pairs: {stats['n_pairs']}")

if __name__ == "__main__":
    main()