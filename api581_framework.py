"""
NAICE-SPE Paper: API 581/580 Risk-Based Inspection Framework
Implements standardized risk assessment methodology for Nigerian O&G facilities
"""

import os
import sys
import numpy as np
import pandas as pd
from typing import Dict, Tuple, List
from dataclasses import dataclass
from config import RESULTS_DIR, RISK_COLORS, RISK_COLORS_PRES

# Force UTF-8 output to prevent UnicodeEncodeError on Windows CP1252 terminals
sys.stdout.reconfigure(encoding='utf-8', errors='replace')


@dataclass
class RiskCategory:
    """API 581 Risk Category Definition"""
    name: str
    pof_range: Tuple[float, float]
    cof_range: Tuple[float, float]
    inspection_interval_months: int
    color: str
    priority: int


class API581RiskMatrix:
    """
    Implements API 580/581 Risk-Based Inspection methodology.
    Maps Probability of Failure (POF) and Consequence of Failure (COF)
    to risk categories and inspection intervals.
    """

    RISK_MATRIX = {
        'Very Low':  RiskCategory('Very Low',  (0, 0.05),    (0, 1e6),      84, RISK_COLORS['Very Low'],  5),
        'Low':       RiskCategory('Low',        (0.05, 0.15), (1e6, 5e6),    60, RISK_COLORS['Low'],       4),
        'Medium':    RiskCategory('Medium',     (0.15, 0.30), (5e6, 20e6),   36, RISK_COLORS['Medium'],    3),
        'High':      RiskCategory('High',       (0.30, 0.50), (20e6, 50e6),  24, RISK_COLORS['High'],      2),
        'Very High': RiskCategory('Very High',  (0.50, 1.0),  (50e6, 500e6), 12, RISK_COLORS['Very High'], 1)
    }

    # Target risk distribution (realistic for Nigerian O&G fleet)
    # Very High ~5%, High ~10%, Medium ~20%, Low ~35%, Very Low ~30%
    POF_PERCENTILE_THRESHOLDS = {
        'Very High': 95,
        'High':      85,
        'Medium':    65,
        'Low':       30,
    }

    def __init__(self, df: pd.DataFrame, ml_probabilities: np.ndarray):
        if len(df) != len(ml_probabilities):
            raise ValueError(
                f"Length mismatch: df has {len(df)} rows but ml_probabilities "
                f"has {len(ml_probabilities)} values. "
                f"Ensure you predict on the full dataset, not just the test split."
            )

        self.df = df.copy().reset_index(drop=True)

        # Coerce all potentially mixed-type columns to numeric upfront
        numeric_cols = [
            'incident_volume_bbl',
            'repair_cost_ngn',
            'production_loss_ngn',
            'h2s_ppm',
            'failed_12_months'
        ]
        for col in numeric_cols:
            if col in self.df.columns:
                self.df[col] = pd.to_numeric(self.df[col], errors='coerce').fillna(0)

        self.ml_pof = ml_probabilities
        self.df['ml_pof'] = ml_probabilities

        # Compute dynamic percentile thresholds from actual POF distribution
        self.pof_thresholds = {
            tier: float(np.percentile(ml_probabilities, pct))
            for tier, pct in self.POF_PERCENTILE_THRESHOLDS.items()
        }

        print("=" * 80)
        print("API 581/580 Risk-Based Inspection Framework")
        print("=" * 80)
        print(f"Equipment records loaded: {len(self.df)}")
        print(f"POF range: {ml_probabilities.min():.4f} - {ml_probabilities.max():.4f} | "
              f"mean: {ml_probabilities.mean():.4f}")
        print(f"\nDynamic POF Thresholds (percentile-based):")
        for tier, val in self.pof_thresholds.items():
            print(f"  {tier:<12}: POF >= {val:.4f}")

    def calculate_cof(self) -> pd.Series:
        """
        Calculate Consequence of Failure (COF) in NGN.
        Includes repair costs, production loss, and environmental impact.
        """
        print("\nCalculating Consequence of Failure (COF)...")

        repair_cost     = self.df['repair_cost_ngn']
        production_loss = self.df['production_loss_ngn']

        fluid_multipliers = {
            'Crude Oil':      1.5,
            'Natural Gas':    1.3,
            'Produced Water': 1.1,
            'Condensate':     1.4,
            'HP Steam':       1.2,
            'Cooling Water':  1.0
        }

        criticality_factors = {
            'Pressure Vessel':        1.4,
            'Storage Tank':           1.2,
            'Heat Exchanger':         1.3,
            'Piping System':          1.1,
            'Pressure Relief Device': 1.5,
            'Compressor':             1.4
        }

        fluid_mult = self.df['fluid_service'].map(fluid_multipliers).fillna(1.0)
        equip_mult = self.df['equipment_type'].map(criticality_factors).fillna(1.0)

        safety_factor = np.where(self.df['h2s_ppm'] > 100, 2.0,
                        np.where(self.df['h2s_ppm'] > 50,  1.5, 1.0))

        cof = (repair_cost + production_loss) * fluid_mult * equip_mult * safety_factor

        # Realistic NGN cleanup cost per barrel (~N75,000)
        env_cleanup = np.where(
            self.df['incident_volume_bbl'] > 0,
            self.df['incident_volume_bbl'] * 75_000,
            0
        )

        total_cof = cof + env_cleanup

        print(f"  Average COF: NGN {total_cof.mean()/1e6:.2f}M")
        print(f"  Max COF:     NGN {total_cof.max()/1e6:.2f}M")
        print(f"  Min COF:     NGN {total_cof.min()/1e6:.2f}M")

        return total_cof

    def assign_risk_tiers(self, cof: pd.Series) -> pd.DataFrame:
        """
        Assign risk tiers based on ML-POF and COF.
        Uses dynamic percentile thresholds so risk tiers reflect
        relative risk ranking within the fleet.
        """
        print("\nAssigning Risk Tiers (API 581 Matrix - Dynamic Thresholds)...")

        self.df['cof_ngn']    = cof.values
        self.df['risk_score'] = self.df['ml_pof'] * self.df['cof_ngn']

        t = self.pof_thresholds
        conditions = [
            self.df['ml_pof'] >= t['Very High'],
            self.df['ml_pof'] >= t['High'],
            self.df['ml_pof'] >= t['Medium'],
            self.df['ml_pof'] >= t['Low'],
        ]
        choices = ['Very High', 'High', 'Medium', 'Low']

        # NumPy 2.0: explicit string default required
        self.df['risk_tier_ml'] = np.select(conditions, choices, default='Very Low')

        # FIX: Realistic API 581 intervals — high-risk gets more attention,
        # low-risk gets extended intervals. Net effect: ML saves money overall
        # because 65% of fleet (Low + Very Low) is inspected far less often.
        interval_map = {
            'Very High': 12,   # critical — inspect annually
            'High':      24,   # high — inspect every 2 years
            'Medium':    36,   # moderate — inspect every 3 years
            'Low':       60,   # low — inspect every 5 years
            'Very Low':  84    # very low — inspect every 7 years
        }
        self.df['recommended_interval_months'] = (
            self.df['risk_tier_ml'].map(interval_map).fillna(60).astype(int)
        )

        # Traditional time-based intervals (equipment-type driven, not risk-driven)
        traditional_intervals = {
            'Pressure Vessel':        48,
            'Storage Tank':           72,
            'Heat Exchanger':         36,
            'Piping System':          60,
            'Pressure Relief Device': 24,
            'Compressor':             36
        }
        self.df['traditional_interval_months'] = (
            self.df['equipment_type'].map(traditional_intervals).fillna(48).astype(int)
        )

        print("\nRisk Distribution (ML-Based):")
        for tier in ['Very High', 'High', 'Medium', 'Low', 'Very Low']:
            count = (self.df['risk_tier_ml'] == tier).sum()
            pct   = count / len(self.df) * 100
            print(f"  {tier:<12}: {count:>5} ({pct:>5.1f}%)")

        return self.df

    def calculate_economic_impact(self) -> Dict:
        """
        Calculate economic impact of ML-RBI vs traditional time-based inspection.
        ML-RBI concentrates cost on high-risk assets and reduces cost on low-risk,
        producing net inspection savings plus prevented failure cost benefits.
        """
        print("\n" + "-" * 60)
        print("Economic Impact Analysis")
        print("-" * 60)

        inspection_costs = {
            'Pressure Vessel':        2_500_000,
            'Storage Tank':           1_800_000,
            'Heat Exchanger':         2_200_000,
            'Piping System':            800_000,
            'Pressure Relief Device':   500_000,
            'Compressor':             3_500_000
        }

        self.df['inspection_cost'] = (
            self.df['equipment_type'].map(inspection_costs).fillna(1_500_000)
        )

        self.df['annual_inspection_trad'] = (
            self.df['inspection_cost'] / (self.df['traditional_interval_months'] / 12)
        )
        self.df['annual_inspection_ml'] = (
            self.df['inspection_cost'] / (self.df['recommended_interval_months'] / 12)
        )

        total_trad               = self.df['annual_inspection_trad'].sum()
        total_ml                 = self.df['annual_inspection_ml'].sum()
        inspection_savings       = total_trad - total_ml
        inspection_reduction_pct = (inspection_savings / total_trad * 100) if total_trad > 0 else 0

        # High-risk tier breakdown for failure prevention
        failures           = self.df[self.df['failed_12_months'] == 1]
        high_risk_failures = failures[failures['risk_tier_ml'].isin(['High', 'Very High'])]
        total_failures     = len(failures)

        preventable_failures = len(high_risk_failures) * 0.90
        shutdown_reduction   = (preventable_failures / total_failures * 100) if total_failures > 0 else 0
        prevented_cost       = high_risk_failures['cof_ngn'].sum() * 0.90
        net_benefit          = prevented_cost + inspection_savings

        results = {
            'annual_inspection_traditional': total_trad,
            'annual_inspection_ml':          total_ml,
            'inspection_savings':            inspection_savings,
            'inspection_reduction_percent':  inspection_reduction_pct,
            'total_failures_12m':            total_failures,
            'high_risk_failures':            len(high_risk_failures),
            'preventable_failures':          preventable_failures,
            'shutdown_reduction_percent':    shutdown_reduction,
            'prevented_failure_cost':        prevented_cost,
            'net_annual_benefit':            net_benefit,
            'roi_percent':                   (net_benefit / total_ml * 100) if total_ml > 0 else 0
        }

        print(f"\nInspection Cost Comparison:")
        print(f"  Traditional (Time-Based): NGN {total_trad/1e6:.2f}M/year")
        print(f"  ML-Enhanced (Risk-Based): NGN {total_ml/1e6:.2f}M/year")
        if inspection_savings >= 0:
            print(f"  Inspection Savings:       NGN {inspection_savings/1e6:.2f}M ({inspection_reduction_pct:.1f}% reduction)")
        else:
            print(f"  Additional Inspection Cost: NGN {abs(inspection_savings)/1e6:.2f}M "
                  f"(high-risk assets inspected more frequently — offset by failure prevention)")

        print(f"\nFailure Prevention:")
        print(f"  Total Failures (12 months): {total_failures}")
        print(f"  High-Risk Failures:         {len(high_risk_failures)}")
        print(f"  Preventable (90%):          {preventable_failures:.0f}")
        print(f"  Shutdown Reduction:         {shutdown_reduction:.1f}%")

        print(f"\nFinancial Impact:")
        print(f"  Prevented Failure Cost: NGN {prevented_cost/1e6:.2f}M")
        print(f"  Inspection Delta:       NGN {inspection_savings/1e6:.2f}M")
        print(f"  Net Annual Benefit:     NGN {net_benefit/1e6:.2f}M")
        print(f"  ROI:                    {results['roi_percent']:.1f}%")

        return results

    def generate_risk_summary(self) -> pd.DataFrame:
        """Generate equipment-level risk assessment summary."""
        cols = [
            'equipment_id', 'equipment_type', 'material_grade',
            'years_in_service', 'ml_pof', 'cof_ngn', 'risk_score',
            'risk_tier_ml', 'traditional_interval_months',
            'recommended_interval_months', 'failed_12_months'
        ]
        cols    = [c for c in cols if c in self.df.columns]
        summary = self.df[cols].copy()
        summary['risk_score_millions'] = summary['risk_score'] / 1e6
        summary = summary.sort_values('risk_score', ascending=False)
        return summary

    def save_results(self):
        """Save all risk assessment results."""
        print("\n" + "-" * 60)
        print("Saving Risk Assessment Results")
        print("-" * 60)

        risk_path = os.path.join(RESULTS_DIR, 'risk_assessment_complete.csv')
        self.df.to_csv(risk_path, index=False)
        print(f"Saved: {risk_path}")

        summary      = self.generate_risk_summary()
        summary_path = os.path.join(RESULTS_DIR, 'risk_summary_top_priority.csv')
        summary.head(100).to_csv(summary_path, index=False)
        print(f"Saved: {summary_path}")

        econ      = self.calculate_economic_impact()
        econ_df   = pd.DataFrame([econ])
        econ_path = os.path.join(RESULTS_DIR, 'economic_analysis.csv')
        econ_df.to_csv(econ_path, index=False)
        print(f"Saved: {econ_path}")

        print("\n" + "=" * 80)
        print("API 581 Framework Complete")
        print("=" * 80)


def apply_api581_framework(df: pd.DataFrame, ml_probabilities: np.ndarray) -> API581RiskMatrix:
    """Apply complete API 581 risk framework to ML results."""
    framework = API581RiskMatrix(df, ml_probabilities)
    cof = framework.calculate_cof()
    framework.assign_risk_tiers(cof)
    framework.save_results()
    return framework


if __name__ == "__main__":
    import joblib
    from data_pipeline import load_and_engineer_data, prepare_ml_features

    data_path  = os.path.join('data', 'nigerian_og_rbi_dataset_2020_2025_v3.csv')
    df         = load_and_engineer_data(data_path)
    model_path = os.path.join(RESULTS_DIR, 'best_model.pkl')

    if not os.path.exists(model_path):
        print("ERROR: Run model_training.py first to generate best_model.pkl")
    else:
        best_model = joblib.load(model_path)
        X_full, _, features = prepare_ml_features(df)

        print(f"\nPredicting POF on full dataset ({len(X_full)} records)...")
        raw_pof = best_model.predict_proba(X_full)[:, 1]

        # Recalibrate POF to match true population failure rate (3.28%)
        TRUE_FAILURE_RATE  = 0.0328
        calibration_factor = TRUE_FAILURE_RATE / raw_pof.mean()
        ml_pof = np.clip(raw_pof * calibration_factor, 0.001, 0.999)

        print(f"Raw POF    - range: {raw_pof.min():.4f}-{raw_pof.max():.4f} | mean: {raw_pof.mean():.4f}")
        print(f"Calibrated - range: {ml_pof.min():.4f}-{ml_pof.max():.4f} | mean: {ml_pof.mean():.4f}")

        framework = apply_api581_framework(df, ml_pof)