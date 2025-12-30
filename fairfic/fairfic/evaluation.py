import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier

from .metrics import compute_group_metrics


def evaluate_model_fairness(model, X, y, groups, test_size=0.3, random_state=42):
    """
    Evaluate model fairness by computing metrics for each group.
    
    Parameters
    ----------
    model : sklearn estimator
        Trained model with predict() and predict_proba() methods.
    X : array-like or DataFrame
        Features.
    y : array-like
        Target labels.
    groups : array-like
        Protected attribute values.
    test_size : float
        Proportion for test split.
    random_state : int
        Random seed.
        
    Returns
    -------
    dict
        Group metrics dictionary.
    tuple
        Test data (X_test, y_test, groups_test, y_pred, y_prob)
    """
    # Split data
    X_train, X_test, y_train, y_test, groups_train, groups_test = train_test_split(
        X, y, groups, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Handle categorical features if X is DataFrame
    if hasattr(X, 'iloc'):
        categorical_cols = X.select_dtypes(include=['object', 'category']).columns.tolist()
        numerical_cols = X.select_dtypes(include=['int64', 'float64']).columns.tolist()
        
        if categorical_cols:
            preprocessor = ColumnTransformer([
                ('num', StandardScaler(), numerical_cols),
                ('cat', OneHotEncoder(drop='first', sparse_output=False), categorical_cols)
            ])
            X_train = preprocessor.fit_transform(X_train)
            X_test = preprocessor.transform(X_test)
    
    # Train model if not already trained
    if not hasattr(model, 'fitted_') or not model.fitted_:
        model.fit(X_train, y_train)
    
    # Make predictions
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
    
    # Compute group metrics
    group_metrics = compute_group_metrics(y_test, y_pred, y_prob, groups_test)
    
    return group_metrics, (X_test, y_test, groups_test, y_pred, y_prob)


def create_default_model(model_type='logistic', **kwargs):
    """
    Create a default model for fairness evaluation.
    
    Parameters
    ----------
    model_type : str
        Type of model: 'logistic', 'random_forest'
    **kwargs : dict
        Model parameters.
        
    Returns
    -------
    sklearn estimator
        Configured model.
    """
    if model_type == 'logistic':
        return LogisticRegression(random_state=42, max_iter=1000, **kwargs)
    elif model_type == 'random_forest':
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(random_state=42, **kwargs)
    elif model_type == 'l1':
        return LogisticRegression(penalty='l1', solver='liblinear', 
                                 random_state=42, max_iter=1000, **kwargs)
    elif model_type == 'l2':
        return LogisticRegression(penalty='l2', random_state=42, 
                                 max_iter=1000, **kwargs)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def analyze_dataset_fairness(X, y, groups, model_type='logistic', 
                            alphaF_values=None, **model_kwargs):
    """
    Complete fairness analysis for a dataset.
    
    Parameters
    ----------
    X : array-like or DataFrame
        Features.
    y : array-like
        Target labels.
    groups : array-like
        Protected attribute.
    model_type : str
        Type of model to use.
    alphaF_values : list
        FIC alphaF values.
    **model_kwargs : dict
        Model parameters.
        
    Returns
    -------
    dict
        Complete analysis results.
    """
    from .core import FairnessInformationCriterion
    
    # Create model
    model = create_default_model(model_type, **model_kwargs)
    
    # Evaluate fairness
    group_metrics, test_data = evaluate_model_fairness(model, X, y, groups)
    
    # Initialize FIC
    if alphaF_values is None:
        alphaF_values = [0.05, 0.10, 0.15, 0.20]
    fic = FairnessInformationCriterion(alphaF_values)
    
    # Analyze for different metrics
    results = {}
    for metric in ['accuracy', 'tpr', 'fpr']:
        if any(metric in gm for gm in group_metrics.values()):
            results[metric] = fic.analyze_fairness(group_metrics, metric)
    
    return {
        'group_metrics': group_metrics,
        'fic_results': results,
        'test_data': test_data,
        'model': model,
        'fic': fic
    }