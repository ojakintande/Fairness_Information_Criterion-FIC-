print("="*60)
print("Testing Fixed metrics.py Import")
print("="*60)

try:
    from fairfic.metrics import compute_all_metrics
    print("✅ Direct import from metrics.py works!")
except ImportError as e:
    print(f"❌ Direct import failed: {e}")

try:
    from fairfic import compute_all_metrics, compute_group_metrics, compute_disparity
    print("✅ Import from fairfic package works!")
    
    # Quick test
    import numpy as np
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 1, 1, 0, 0])
    y_prob = np.array([0.1, 0.9, 0.6, 0.8, 0.2, 0.4])
    groups = np.array(['A', 'A', 'B', 'B', 'A', 'B'])
    
    # Test compute_all_metrics
    metrics = compute_all_metrics(y_true, y_pred, y_prob)
    print(f"✅ compute_all_metrics works! Got {len(metrics)} metrics")
    
    # Test compute_group_metrics
    group_metrics = compute_group_metrics(y_true, y_pred, y_prob, groups)
    print(f"✅ compute_group_metrics works! Got {len(group_metrics)} groups")
    
    # Test compute_disparity
    disparity = compute_disparity(group_metrics, 'accuracy')
    print(f"✅ compute_disparity works! Disparity = {disparity['disparity']:.3f}")
    
except ImportError as e:
    print(f"❌ Package import failed: {e}")
except Exception as e:
    print(f"❌ Other error: {type(e).__name__}: {e}")

print("="*60)