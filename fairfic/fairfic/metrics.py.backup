import numpy as np
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    confusion_matrix, precision_score, recall_score
)


def compute_all_metrics(y_true, y_pred, y_prob=None):
    """
    Compute comprehensive performance metrics.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like, optional
        Predicted probabilities for positive class.
        
    Returns
    -------
    dict
        Dictionary containing all computed metrics.
    """
    # Compute confusion matrix
    tn, fp, fn, tp = confusion_matrix(y_true, y_pred).ravel()
    
    # Base metrics
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, zero_division=0),
        'recall': recall_score(y_true, y_pred, zero_division=0),
        'f1': f1_score(y_true, y_pred, zero_division=0),
        'selection_rate': (tp + fp) / len(y_true) if len(y_true) > 0 else 0,
    }
    
    # Rates from confusion matrix
    metrics.update({
        'tpr': tp / (tp + fn) if (tp + fn) > 0 else 0,      # True Positive Rate (Recall)
        'tnr': tn / (tn + fp) if (tn + fp) > 0 else 0,      # True Negative Rate
        'fpr': fp / (fp + tn) if (fp + tn) > 0 else 0,      # False Positive Rate
        'fnr': fn / (tp + fn) if (tp + fn) > 0 else 0,      # False Negative Rate
        'ppv': tp / (tp + fp) if (tp + fp) > 0 else 0,      # Positive Predictive Value (Precision)
        'npv': tn / (tn + fn) if (tn + fn) > 0 else 0,      # Negative Predictive Value
    })
    
    # AUC if probabilities are provided
    if y_prob is not None and len(np.unique(y_true)) > 1:
        try:
            metrics['auc'] = roc_auc_score(y_true, y_prob)
        except:
            metrics['auc'] = np.nan
    else:
        metrics['auc'] = np.nan
    
    return metrics


def compute_group_metrics(y_true, y_pred, y_prob, groups):
    """
    Compute metrics for each group separately.
    
    Parameters
    ----------
    y_true : array-like
        Ground truth labels.
    y_pred : array-like
        Predicted labels.
    y_prob : array-like
        Predicted probabilities.
    groups : array-like
        Group membership for each sample.
        
    Returns
    -------
    dict
        Dictionary mapping group names to metric dictionaries.
    """
    unique_groups = np.unique(groups)
    group_metrics = {}
    
    for group in unique_groups:
        mask = groups == group
        if np.sum(mask) > 0:
            group_metrics[group] = compute_all_metrics(
                y_true[mask], y_pred[mask], 
                y_prob[mask] if y_prob is not None else None
            )
    
    return group_metrics


def compute_disparity(group_metrics, metric_name='accuracy'):
    """
    Compute disparity (maximum difference) for a specific metric across groups.
    
    Parameters
    ----------
    group_metrics : dict
        Dictionary of metrics by group.
    metric_name : str
        Name of metric to compute disparity for.
        
    Returns
    -------
    dict
        Dictionary containing disparity metrics.
    """
    values = []
    valid_groups = []
    
    for group, metrics in group_metrics.items():
        if metric_name in metrics and not np.isnan(metrics[metric_name]):
            values.append(metrics[metric_name])
            valid_groups.append(group)
    
    if len(values) < 2:
        return {
            'disparity': np.nan,
            'min_value': np.nan,
            'max_value': np.nan,
            'min_group': '',
            'max_group': '',
            'n_groups': len(values)
        }
    
    min_idx = np.argmin(values)
    max_idx = np.argmax(values)
    
    return {
        'disparity': values[max_idx] - values[min_idx],
        'min_value': values[min_idx],
        'max_value': values[max_idx],
        'min_group': valid_groups[min_idx],
        'max_group': valid_groups[max_idx],
        'n_groups': len(values)
    }