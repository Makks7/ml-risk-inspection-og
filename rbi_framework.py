
"""
NAICE PAPER: ML-ENHANCED RBI FOR NIGERIAN O&G
"""

import numpy as np
import pandas as pd
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

print("=" * 70)
print("NAICE PAPER: ML-ENHANCED RBI")
print("=" * 70)

# Load data
df = pd.read_csv('nigerian_og_rbi_dataset.csv')
print("Loaded: {} records".format(len(df)))

# Save original columns needed later
df['equipment_type_original'] = df['equipment_type']
df['repair_cost'] = df['repair_cost_ngn']
df['production_loss'] = df['production_loss_ngn']

# Feature engineering
df['age_category'] = pd.cut(df['years_in_service'], bins=[0, 5, 10, 20, 50], labels=['New', 'Medium', 'Aged', 'Old'])
df['env_stress_score'] = (df['h2s_ppm'] / 500 + df['co2_partial_pressure_bar'] / 50 + df['humidity_percent'] / 100) / 3
df['wall_loss_ratio'] = df['wall_loss_mm'] / df['wall_thickness_mm']
df['last_known_thickness'] = df['wall_thickness_mm'] - df['wall_loss_mm']
df['operating_severity'] = (df['operating_pressure_bar'] / 150 + df['operating_temp_c'] / 400) / 2

material_vuln = {'Carbon Steel': 1.0, 'Stainless Steel 304': 0.3, 'Stainless Steel 316': 0.2, 'Duplex Stainless': 0.15, 'Inconel 625': 0.05}
df['material_vulnerability'] = df['material_grade'].map(material_vuln)
df['time_since_inspection_years'] = np.random.uniform(0, 5, len(df))
df['inspection_quality_score'] = np.random.uniform(0.5, 1.0, len(df))
df['dm_thinning'] = (df['corrosion_rate_mm_yr'] > 0.1).astype(int)
df['dm_scc'] = ((df['h2s_ppm'] > 50) & (df['material_grade'] == 'Carbon Steel')).astype(int)
df['dm_external'] = (df['salt_exposure_factor'] > 1.5).astype(int)

# One-hot encode
df = pd.get_dummies(df, columns=['equipment_type', 'material_grade', 'facility_location', 'age_category'], prefix=['type', 'mat', 'loc', 'age'])

# Features and target
exclude = ['equipment_id', 'installation_date', 'failure_date', 'failure_mode', 'calculated_pof', 
           'repair_cost_ngn', 'production_loss_ngn', 'preventable', 'failed_12_months', 
           'wall_loss_mm', 'wall_thickness_mm', 'equipment_type_original', 'repair_cost', 'production_loss']
features = [c for c in df.columns if c not in exclude]
X = df[features]
y = df['failed_12_months']

# Temporal split
df_sorted = df.sort_values('years_in_service')
X_sorted = df_sorted[features]
y_sorted = df_sorted['failed_12_months']

n = len(df_sorted)
X_train, X_val, X_test = X_sorted.iloc[:int(n*0.7)], X_sorted.iloc[int(n*0.7):int(n*0.85)], X_sorted.iloc[int(n*0.85):]
y_train, y_val, y_test = y_sorted.iloc[:int(n*0.7)], y_sorted.iloc[int(n*0.7):int(n*0.85)], y_sorted.iloc[int(n*0.85):]

print("Split: Train {} | Val {} | Test {}".format(len(X_train), len(X_val), len(X_test)))

# Train model
print("\nTraining Random Forest...")
rf = RandomForestClassifier(n_estimators=300, max_depth=12, class_weight='balanced_subsample', random_state=42, n_jobs=-1)
rf.fit(X_train, y_train)

# Predict
y_proba = rf.predict_proba(X_test)[:, 1]
y_pred = rf.predict(X_test)

# Metrics
auc = roc_auc_score(y_test, y_proba)
ap = average_precision_score(y_test, y_proba)
f1 = f1_score(y_test, y_pred)

print("\nResults:")
print("  ROC-AUC: {:.4f}".format(auc))
print("  Avg Precision: {:.4f}".format(ap))
print("  F1-Score: {:.4f}".format(f1))

# Economic analysis
df_test = df_sorted.iloc[int(n*0.85):].copy()
df_test['ml_proba'] = y_proba
df_test['risk_quintile'] = pd.qcut(df_test['ml_proba'], 5, labels=['Very Low', 'Low', 'Medium', 'High', 'Very High'])

# Costs - use original saved column
equip_costs = {'Pressure Vessel': 2500000, 'Storage Tank': 1800000, 'Heat Exchanger': 2200000, 
               'Piping System': 800000, 'Pressure Relief Device': 500000, 'Compressor': 3500000}
trad_intervals = {'Pressure Vessel': 4, 'Storage Tank': 6, 'Heat Exchanger': 3, 
                  'Piping System': 5, 'Pressure Relief Device': 2, 'Compressor': 3}

df_test['inspection_cost'] = df_test['equipment_type_original'].map(equip_costs)
df_test['trad_interval'] = df_test['equipment_type_original'].map(trad_intervals)

ml_intervals = {'Very Low': 8, 'Low': 6, 'Medium': 4, 'High': 2, 'Very High': 1}
df_test['ml_interval'] = df_test['risk_quintile'].map(ml_intervals).astype(float)

trad_insp = (df_test['inspection_cost'] / df_test['trad_interval']).sum()
ml_insp = (df_test['inspection_cost'] / df_test['ml_interval']).sum()

# Failure prevention
total_failures = df_test['failed_12_months'].sum()
high_risk = df_test[(df_test['risk_quintile'].isin(['High', 'Very High'])) & (df_test['failed_12_months'] == 1)]
prevented = len(high_risk) * 0.90
shutdown_red = (prevented / total_failures) * 100

trad_fail = df_test[df_test['failed_12_months']==1]['repair_cost'].sum() + df_test[df_test['failed_12_months']==1]['production_loss'].sum()
ml_fail = trad_fail * ((total_failures - prevented) / total_failures)

trad_total = trad_insp + trad_fail
ml_total = ml_insp + ml_fail
cost_red = (trad_total - ml_total) / trad_total * 100

print("\nEconomic Impact:")
print("  Traditional cost: NGN {:.1f}M".format(trad_total/1e6))
print("  ML cost: NGN {:.1f}M".format(ml_total/1e6))
print("  Cost reduction: {:.1f}%".format(cost_red))
print("  Shutdown reduction: {:.1f}%".format(shutdown_red))

# Save everything
df_test.to_csv('final_results.csv', index=False)
joblib.dump(rf, 'model.pkl')
pd.Series(features).to_csv('features.csv', index=False)
np.save('y_test.npy', y_test.values)
np.save('y_proba.npy', y_proba)

metrics = {
    'auc': auc, 'ap': ap, 'f1': f1,
    'inspection_trad': trad_insp/1e6, 'inspection_ml': ml_insp/1e6,
    'failure_trad': trad_fail/1e6, 'failure_ml': ml_fail/1e6,
    'total_trad': trad_total/1e6, 'total_ml': ml_total/1e6,
    'failures_trad': total_failures, 'failures_ml': total_failures - prevented,
    'shutdown_reduction': shutdown_red, 'cost_reduction': cost_red
}
pd.Series(metrics).to_csv('metrics.csv')

print("\n[OK] Done! Files saved.")
print("Run: python generate_plots.py")