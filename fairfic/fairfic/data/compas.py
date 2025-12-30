import pandas as pd
import numpy as np
import os


def load_compas_data(data_path=None, sample_size=None, random_state=42):
    """
    Load and preprocess COMPAS dataset.
    
    Parameters
    ----------
    data_path : str, optional
        Path to COMPAS CSV file. If None, tries default locations.
    sample_size : int, optional
        Number of samples to return (random sample).
    random_state : int
        Random seed for sampling.
        
    Returns
    -------
    pandas.DataFrame
        Preprocessed COMPAS dataset.
        
    Notes
    -----
    The COMPAS dataset should have columns including:
    - 'race', 'age', 'sex', 'priors_count', 'c_charge_degree'
    - 'decile_score', 'two_year_recid'
    """
    # If no path provided, try to find it
    if data_path is None:
        # Try common locations
        possible_paths = [
            "compas-scores-two-years.csv",
            "data/compas-scores-two-years.csv",
            "../data/compas-scores-two-years.csv",
        ]
        
        for path in possible_paths:
            if os.path.exists(path):
                data_path = path
                break
        
        if data_path is None:
            raise FileNotFoundError(
                "COMPAS dataset not found. Please download it from: "
                "https://github.com/propublica/compas-analysis/raw/master/compas-scores-two-years.csv"
            )
    
    print(f"Loading COMPAS dataset from: {data_path}")
    df = pd.read_csv(data_path)
    
    # Select relevant columns
    relevant_columns = [
        'age', 'sex', 'race', 'priors_count', 'c_charge_degree',
        'juv_fel_count', 'juv_misd_count', 'juv_other_count',
        'decile_score', 'two_year_recid'
    ]
    
    # Keep only available columns
    available_columns = [col for col in relevant_columns if col in df.columns]
    df = df[available_columns].copy()
    
    # Drop missing values
    df = df.dropna()
    
    # Create target: high risk if decile_score >= 6
    df['high_risk'] = (df['decile_score'] >= 6).astype(int)
    
    # Consolidate race categories
    def consolidate_race(race):
        race = str(race).strip().lower()
        if 'african' in race or 'black' in race:
            return 'African_American'
        elif 'caucasian' in race or 'white' in race:
            return 'Caucasian'
        elif 'hispanic' in race or 'latino' in race:
            return 'Hispanic'
        else:
            return 'Other'
    
    df['race_group'] = df['race'].apply(consolidate_race)
    
    # Filter to keep only major race groups
    target_races = ['African_American', 'Caucasian', 'Hispanic']
    df = df[df['race_group'].isin(target_races)].copy()
    
    # Create additional features
    df['total_juvenile_charges'] = df['juv_fel_count'] + df['juv_misd_count'] + df['juv_other_count']
    df['is_felony'] = (df['c_charge_degree'] == 'F').astype(int)
    
    # Create age groups
    df['age_group'] = pd.cut(df['age'], 
                             bins=[0, 25, 35, 45, 55, 100],
                             labels=['18-25', '26-35', '36-45', '46-55', '56+'])
    
    # Select final columns
    final_columns = [
        'age', 'sex', 'race_group', 'priors_count', 'is_felony',
        'total_juvenile_charges', 'age_group', 'high_risk'
    ]
    
    # Ensure all columns exist
    final_columns = [col for col in final_columns if col in df.columns]
    df = df[final_columns]
    
    # Sample if requested
    if sample_size and sample_size < len(df):
        df = df.sample(n=sample_size, random_state=random_state)
    
    print(f"Processed dataset shape: {df.shape}")
    print(f"Target distribution (high_risk):")
    print(df['high_risk'].value_counts(normalize=True).round(3))
    print(f"\nRace group distribution:")
    print(df['race_group'].value_counts(normalize=True).round(3))
    
    return df


def get_compas_features_targets(df, target_col='high_risk', protected_col='race_group'):
    """
    Extract features and targets from COMPAS dataframe.
    
    Parameters
    ----------
    df : pandas.DataFrame
        COMPAS dataframe from load_compas_data().
    target_col : str
        Name of target column.
    protected_col : str
        Name of protected attribute column.
        
    Returns
    -------
    tuple
        (X, y, protected_attr)
    """
    X = df.drop(columns=[target_col, protected_col])
    y = df[target_col]
    protected_attr = df[protected_col]
    
    return X, y, protected_attr