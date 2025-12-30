"""
Core Fairness Information Criterion (FIC) implementation.
"""

import numpy as np


class FairnessInformationCriterion:
    """
    Fairness Information Criterion (FIC) for evaluating algorithmic fairness.
    
    FIC measures fairness between groups by comparing performance metrics
    and classifying fairness into tiers based on a threshold parameter alphaF.
    
    Parameters
    ----------
    alphaF_values : list of float, default=[0.05, 0.10, 0.15, 0.20]
        List of fairness thresholds. Lower alphaF means stricter fairness criteria.
    
    Attributes
    ----------
    alphaF_values : list of float
        Fairness thresholds used for evaluation.
    
    Examples
    --------
    >>> from fairfic import FairnessInformationCriterion
    >>> fic = FairnessInformationCriterion(alphaF_values=[0.05, 0.10])
    >>> group_metrics = {
    ...     'Group_A': {'accuracy': 0.85, 'tpr': 0.80},
    ...     'Group_B': {'accuracy': 0.78, 'tpr': 0.75}
    ... }
    >>> results = fic.analyze_fairness(group_metrics, metric_name='accuracy')
    """
    
    def __init__(self, alphaF_values=None):
        """
        Initialize the Fairness Information Criterion.
        
        Parameters
        ----------
        alphaF_values : list of float, optional
            List of fairness thresholds. Default is [0.05, 0.10, 0.15, 0.20].
        """
        if alphaF_values is None:
            alphaF_values = [0.05, 0.10, 0.15, 0.20]
        self.alphaF_values = alphaF_values
    
    def compute_omega(self, metric1, metric2):
        """
        Compute the absolute difference between two metric values.
        
        Parameters
        ----------
        metric1 : float
            Metric value for first group.
        metric2 : float
            Metric value for second group.
        
        Returns
        -------
        float
            Absolute difference: |metric1 - metric2|
        """
        return abs(metric1 - metric2)
    
    def compute_fic(self, omega, alphaF):
        """
        Compute the Fairness Information Criterion (FIC) score.
        
        FIC = 1 - (omega / alphaF)
        
        Parameters
        ----------
        omega : float
            Absolute difference between group metrics.
        alphaF : float
            Fairness threshold parameter.
        
        Returns
        -------
        float
            FIC score. Higher values indicate better fairness.
        """
        if alphaF <= 0:
            raise ValueError("alphaF must be positive")
        return 1 - (omega / alphaF)
    
    def classify_tier(self, fic_score):
        """
        Classify fairness into tiers based on FIC score.
        
        Tiers:
        - Optimum: FIC > 0.75
        - Acceptable: 0.50 < FIC ≤ 0.75
        - Questionable: 0 < FIC ≤ 0.50
        - Unacceptable: FIC ≤ 0
        
        Parameters
        ----------
        fic_score : float
            FIC score to classify.
        
        Returns
        -------
        str
            Tier classification.
        """
        if fic_score > 0.75:
            return "Optimum"
        elif fic_score > 0.50:
            return "Acceptable"
        elif fic_score > 0:
            return "Questionable"
        else:
            return "Unacceptable"
    
    def analyze_fairness(self, group_metrics, metric_name='accuracy'):
        """
        Analyze fairness across groups for a specific metric.
        
        Parameters
        ----------
        group_metrics : dict
            Dictionary mapping group names to metric dictionaries.
            Example: {'Group_A': {'accuracy': 0.8}, 'Group_B': {'accuracy': 0.75}}
        metric_name : str, default='accuracy'
            Name of the metric to analyze.
        
        Returns
        -------
        dict
            Nested dictionary with results for each alphaF value.
            Structure: {alphaF: {group_pair: {results}}}
        """
        results = {}
        groups = list(group_metrics.keys())
        
        for alphaF in self.alphaF_values:
            results[alphaF] = {}
            
            # Compare all unique pairs of groups
            for i, g1 in enumerate(groups):
                for g2 in groups[i+1:]:
                    pair = f"{g1} - {g2}"
                    
                    # Get metric values
                    m1 = group_metrics[g1].get(metric_name, np.nan)
                    m2 = group_metrics[g2].get(metric_name, np.nan)
                    
                    # Skip if either metric is missing
                    if np.isnan(m1) or np.isnan(m2):
                        continue
                    
                    # Compute FIC
                    omega = self.compute_omega(m1, m2)
                    fic_score = self.compute_fic(omega, alphaF)
                    tier = self.classify_tier(fic_score)
                    
                    results[alphaF][pair] = {
                        'omega': omega,
                        'fic_score': fic_score,
                        'tier': tier,
                        f'{g1}_value': m1,
                        f'{g2}_value': m2
                    }
        
        return results
    
    def get_fairness_summary(self, fic_results):
        """
        Generate a summary of fairness analysis results.
        
        Parameters
        ----------
        fic_results : dict
            Results from analyze_fairness() method.
        
        Returns
        -------
        dict
            Summary statistics for each alphaF value.
        """
        summary = {}
        
        for alphaF, pairs_data in fic_results.items():
            if not pairs_data:
                summary[alphaF] = {'message': 'No data available'}
                continue
            
            omegas = [d['omega'] for d in pairs_data.values()]
            fic_scores = [d['fic_score'] for d in pairs_data.values()]
            tiers = [d['tier'] for d in pairs_data.values()]
            
            # Count tiers
            tier_counts = {
                'Optimum': tiers.count('Optimum'),
                'Acceptable': tiers.count('Acceptable'),
                'Questionable': tiers.count('Questionable'),
                'Unacceptable': tiers.count('Unacceptable')
            }
            
            summary[alphaF] = {
                'omega_mean': np.mean(omegas),
                'omega_std': np.std(omegas),
                'omega_min': np.min(omegas),
                'omega_max': np.max(omegas),
                'fic_mean': np.mean(fic_scores),
                'fic_min': np.min(fic_scores),
                'fic_max': np.max(fic_scores),
                'tier_counts': tier_counts,
                'n_pairs': len(pairs_data)
            }
        
        return summary