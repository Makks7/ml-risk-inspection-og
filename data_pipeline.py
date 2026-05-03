"""
NAICE-SPE Paper: Data Pipeline & Feature Engineering
Processes Nigerian O&G RBI Dataset for ML Analysis
"""

import pandas as pd
import numpy as np
import os
from config import DATA_DIR, RESULTS_DIR

def load_and_engineer_data(filepath):
    """
    Load Nigerian O&G RBI dataset and engineer features for ML model
    
    Parameters:
    -----------
    filepath : str
        Path to nigerian_og_rbi_dataset_2020_2025_v3.csv
        
    Returns:
    --------
    df : pandas.DataFrame
        Processed dataframe with engineered features
    """
    
    print("=" * 70)
    print("NAICE-SPE: Data Pipeline & Feature Engineering")
    print("=" * 70)
    
    # Load data
    df = pd.read_csv(filepath)
    print(f"\nLoaded dataset: {df.shape[0]} records, {df.shape[1]} features")
    
    # Store original identifiers
    df['equipment_id'] = df['equipment_id']
    df['equipment_type_original'] = df['equipment_type']
    df['material_original'] = df['material_grade']
    
    # Target variable
    target_col = 'failed_12_months'
    print(f"\nTarget distribution:")
    print(f"  Failures: {df[target_col].sum()} ({df[target_col].mean()*100:.2f}%)")
    print(f"  No failure: {(df[target_col]==0).sum()} ({(1-df[target_col].mean())*100:.2f}%)")
    
    # ==========================================
    # FEATURE ENGINEERING - Nigerian O&G Context
    # ==========================================
    
    print("\n" + "-" * 50)
    print("ENGINEERING FEATURES")
    print("-" * 50)
    
    # 1. Temporal Features
    df['age_category'] = pd.cut(
        df['years_in_service'], 
        bins=[0, 5, 10, 20, 50], 
        labels=['New', 'Medium', 'Aged', 'Old']
    )
    df['years_since_inspection'] = df['last_inspection_yrs_ago']
    
    # 2. Environmental Stress Index (Critical for Nigerian climate)
    df['env_stress_score'] = (
        (df['h2s_ppm'] / 500) + 
        (df['co2_partial_pressure_bar'] / 50) + 
        (df['humidity_percent'] / 100) + 
        (df['salt_exposure_factor'] / 3)
    ) / 4
    
    # 3. Mechanical Integrity Features
    df['wall_loss_ratio'] = df['wall_loss_mm'] / df['wall_thickness_mm']
    df['wall_integrity_ratio'] = 1 - df['wall_loss_ratio']
    df['remaining_wall_mm'] = df['wall_thickness_mm'] - df['wall_loss_mm']
    
    # 4. Operating Severity (Pressure-Temperature interaction)
    df['operating_severity'] = (
        (df['operating_pressure_bar'] / 200) + 
        (df['operating_temp_c'] / 400)
    ) / 2
    
    # 5. Material Vulnerability Mapping
    material_vuln = {
        'Carbon Steel': 1.0,
        'Stainless Steel 304': 0.3,
        'Stainless Steel 316': 0.2,
        'Duplex Stainless': 0.15,
        'Inconel 625': 0.05
    }
    df['material_vulnerability'] = df['material_grade'].map(material_vuln)
    
    # 6. Damage Mechanism Indicators (API 581 aligned)
    df['dm_thinning'] = (df['corrosion_rate_adj_mm_yr'] > 0.1).astype(int)
    df['dm_scc'] = (
        (df['h2s_ppm'] > 50) & 
        (df['material_grade'] == 'Carbon Steel')
    ).astype(int)
    df['dm_external_corrosion'] = (df['salt_exposure_factor'] > 1.5).astype(int)
    df['dm_sour_service'] = df['is_sour_service'].astype(int)
    
    # 7. Protection Effectiveness
    df['protection_score'] = (
        df['inhibitor_applied'] * 0.6 + 
        df['cathodic_protection'] * 0.4
    )
    
    # 8. Inspection Effectiveness Score
    inspection_map = {'A': 5, 'B': 4, 'C': 3, 'D': 2, 'E': 1}
    df['inspection_effectiveness_score'] = df['inspection_effectiveness'].map(inspection_map)
    
    # 9. Cumulative Risk Score
    df['cumulative_risk'] = (
        df['years_in_service'] * 0.05 +
        df['wall_loss_ratio'] * 2.0 +
        df['env_stress_score'] * 1.5 +
        df['operating_severity'] * 1.0 -
        df['protection_score'] * 0.5 +
        df['dm_thinning'] * 0.8 +
        df['dm_scc'] * 1.2
    )
    
    print(f"Features engineered: {df.shape[1]} total columns")
    
    # Save processed data
    output_path = os.path.join(RESULTS_DIR, 'processed_data.csv')
    df.to_csv(output_path, index=False)
    print(f"\nProcessed data saved to: {output_path}")
    
    return df

def prepare_ml_features(df):
    """
    Prepare final feature matrix for ML modeling
    
    Returns:
    --------
    X : pandas.DataFrame
        Feature matrix
    y : pandas.Series
        Target vector
    feature_names : list
        List of feature names
    """
    
    # Columns to exclude from modeling
    exclude_cols = [
        'equipment_id', 'equipment_type_original', 'material_original',
        'installation_date', 'failure_date', 'failure_mode', 
        'calculated_pof', 'failed_12_months', 'failure_confirmed_jiv',
        'repair_cost_ngn', 'production_loss_ngn', 'risk_score_ngn_M',
        'preventable', 'incident_volume_bbl', 'jiv_reference',
        'wall_loss_mm', 'wall_thickness_mm', 'record_date',
        'operator', 'contract_type', 'terrain', 'fluid_service',
        'facility_location', 'naira_usd_rate', 'data_tier',
        'age_category', 'risk_tier', 'recommended_insp_interval'
    ]
    
    # Identify numeric and categorical features
    numeric_features = [
        'years_in_service', 'operating_pressure_bar', 'operating_temp_c',
        'humidity_percent', 'salt_exposure_factor', 'h2s_ppm',
        'co2_partial_pressure_bar', 'years_since_inspection',
        'corrosion_rate_mm_yr', 'corrosion_rate_adj_mm_yr',
        'wall_integrity_ratio', 'remaining_wall_mm',
        'env_stress_score', 'operating_severity', 'material_vulnerability',
        'dm_thinning', 'dm_scc', 'dm_external_corrosion', 'dm_sour_service',
        'protection_score', 'inspection_effectiveness_score', 'cumulative_risk',
        'n_past_inspections'
    ]
    
    # One-hot encode categorical variables
    categorical_cols = ['equipment_type', 'material_grade']
    df_encoded = pd.get_dummies(df, columns=categorical_cols, prefix=['type', 'mat'])
    
    # Get dummy column names
    dummy_cols = [c for c in df_encoded.columns if c.startswith(('type_', 'mat_'))]
    
    # Final feature set
    feature_cols = numeric_features + dummy_cols
    
    # Filter to available columns only
    available_features = [c for c in feature_cols if c in df_encoded.columns]
    
    X = df_encoded[available_features]
    y = df['failed_12_months']
    
    # Handle missing values
    X = X.fillna(X.median())
    
    print(f"\nFinal ML feature set: {len(available_features)} features")
    print(f"Feature matrix shape: {X.shape}")
    
    return X, y, available_features

if __name__ == "__main__":
    # Example usage
    data_path = os.path.join(DATA_DIR, 'nigerian_og_rbi_dataset_2020_2025_v3.csv')
    df = load_and_engineer_data(data_path)
    X, y, features = prepare_ml_features(df)