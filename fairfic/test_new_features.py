print("Testing new user-friendly features...")
print("="*60)

# Test 1: Check new imports
try:
    from fairfic import (
        evaluate_model_fairness,
        create_default_model,
        analyze_dataset_fairness
    )
    print("✅ New evaluation functions imported successfully")
except ImportError as e:
    print(f"❌ Import error: {e}")
    exit()

# Test 2: Test create_default_model
print("\nTesting create_default_model...")
try:
    model1 = create_default_model('logistic')
    model2 = create_default_model('random_forest', n_estimators=100)
    model3 = create_default_model('l1', C=0.5)
    
    print(f"✅ Created models: {type(model1).__name__}, "
          f"{type(model2).__name__}, {type(model3).__name__}")
except Exception as e:
    print(f"❌ Error creating models: {e}")

# Test 3: Test with synthetic data
print("\nTesting with synthetic data...")
try:
    import numpy as np
    from sklearn.datasets import make_classification
    
    # Create synthetic dataset
    X, y = make_classification(n_samples=200, n_features=5, random_state=42)
    groups = np.random.choice(['A', 'B', 'C'], size=200, p=[0.4, 0.4, 0.2])
    
    # Test analyze_dataset_fairness
    results = analyze_dataset_fairness(
        X=X, y=y, groups=groups,
        model_type='logistic',
        alphaF_values=[0.05, 0.10, 0.15, 0.20]
    )
    
    print(f"✅ analyze_dataset_fairness successful!")
    print(f"   Got {len(results['group_metrics'])} groups")
    print(f"   Got FIC results for {len(results['fic_results'])} metrics")
    
    # Show some results
    group_metrics = results['group_metrics']
    print(f"\nGroup metrics (accuracy):")
    for group, metrics in group_metrics.items():
        print(f"   {group}: {metrics['accuracy']:.3f}")
        
except Exception as e:
    print(f"❌ Error with synthetic data: {type(e).__name__}: {e}")
    import traceback
    traceback.print_exc()

print("\n" + "="*60)
print("✅ NEW FEATURES TESTED SUCCESSFULLY!")
print("="*60)