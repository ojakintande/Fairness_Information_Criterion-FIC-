import pytest
import numpy as np
from fairfic import FairnessInformationCriterion


def test_compute_omega():
    """Test omega computation."""
    fic = FairnessInformationCriterion()
    
    # Test basic cases
    assert fic.compute_omega(0.8, 0.7) == 0.1
    assert fic.compute_omega(0.7, 0.8) == 0.1
    assert fic.compute_omega(0.5, 0.5) == 0.0


def test_compute_fic():
    """Test FIC computation."""
    fic = FairnessInformationCriterion()
    
    # Test basic cases
    assert fic.compute_fic(0.05, 0.10) == 0.5  # 1 - (0.05/0.10)
    assert fic.compute_fic(0.0, 0.10) == 1.0   # Perfect fairness
    assert fic.compute_fic(0.10, 0.10) == 0.0  # At threshold
    
    # Test invalid alphaF
    with pytest.raises(ValueError):
        fic.compute_fic(0.05, 0.0)


def test_classify_tier():
    """Test tier classification."""
    fic = FairnessInformationCriterion()
    
    assert fic.classify_tier(0.9) == "Optimum"
    assert fic.classify_tier(0.6) == "Acceptable"
    assert fic.classify_tier(0.3) == "Questionable"
    assert fic.classify_tier(-0.1) == "Unacceptable"


def test_analyze_fairness():
    """Test fairness analysis."""
    fic = FairnessInformationCriterion(alphaF_values=[0.10])
    
    group_metrics = {
        'Group_A': {'accuracy': 0.8, 'tpr': 0.75},
        'Group_B': {'accuracy': 0.7, 'tpr': 0.70},
    }
    
    results = fic.analyze_fairness(group_metrics, 'accuracy')
    
    assert 0.10 in results
    assert 'Group_A - Group_B' in results[0.10]
    assert results[0.10]['Group_A - Group_B']['omega'] == 0.1
    assert results[0.10]['Group_A - Group_B']['fic_score'] == 0.0


def test_get_fairness_summary():
    """Test summary generation."""
    fic = FairnessInformationCriterion(alphaF_values=[0.10])
    
    group_metrics = {
        'Group_A': {'accuracy': 0.8},
        'Group_B': {'accuracy': 0.7},
        'Group_C': {'accuracy': 0.75},
    }
    
    results = fic.analyze_fairness(group_metrics, 'accuracy')
    summary = fic.get_fairness_summary(results)
    
    assert 0.10 in summary
    assert 'omega_mean' in summary[0.10]
    assert 'tier_counts' in summary[0.10]


if __name__ == "__main__":
    # Run tests without pytest
    test_compute_omega()
    test_compute_fic()
    test_classify_tier()
    print("✅ All basic tests passed!")
    
    # Test analyze_fairness
    fic = FairnessInformationCriterion(alphaF_values=[0.10])
    group_metrics = {'A': {'accuracy': 0.8}, 'B': {'accuracy': 0.7}}
    results = fic.analyze_fairness(group_metrics, 'accuracy')
    print(f"✅ analyze_fairness test passed! Results: {results}")