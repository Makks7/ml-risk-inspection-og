"""
NAICE-SPE Paper: Publication-Quality Figure Generation
Generates all figures for manuscript and conference presentation
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpec
import seaborn as sns
from sklearn.metrics import roc_curve, precision_recall_curve, auc
from scipy import stats

# Force UTF-8 output on Windows
sys.stdout.reconfigure(encoding='utf-8', errors='replace')

from config import (
    FIGURES_PAPER_DIR, FIGURES_PRESENTATION_DIR,
    set_paper_style, set_presentation_style, RISK_COLORS, RISK_COLORS_PRES
)


class FigureGenerator:
    """
    Generates publication-quality figures for NAICE-SPE paper and presentation.
    All figures follow SPE formatting guidelines and API 581 color standards.
    """

    def __init__(self, results_dir: str = 'results'):
        self.results_dir = results_dir
        self.figures_paper = []
        self.figures_presentation = []
        self._load_data()

    def _load_data(self):
        """Load all necessary results files."""
        print("Loading results for figure generation...")

        pred_path = os.path.join(self.results_dir, 'predictions.csv')
        if os.path.exists(pred_path):
            self.predictions = pd.read_csv(pred_path)
        else:
            raise FileNotFoundError("Run model_training.py first")

        imp_path = os.path.join(self.results_dir, 'feature_importance.csv')
        self.feature_importance = pd.read_csv(imp_path) if os.path.exists(imp_path) else None

        risk_path = os.path.join(self.results_dir, 'risk_assessment_complete.csv')
        self.risk_data = pd.read_csv(risk_path) if os.path.exists(risk_path) else None

        comp_path = os.path.join(self.results_dir, 'model_comparison.csv')
        self.model_comparison = pd.read_csv(comp_path) if os.path.exists(comp_path) else None

        print("Data loaded successfully")

    # =========================================================================
    # PAPER FIGURES
    # =========================================================================

    def fig1_feature_importance(self):
        """Figure 1: Top 15 Feature Importance (horizontal bar)"""
        set_paper_style()

        if self.feature_importance is None:
            print("Feature importance data not available")
            return

        fig, ax = plt.subplots(figsize=(8, 6))

        top_features = self.feature_importance.head(15).sort_values('Importance_Relative')
        colors = plt.cm.RdYlGn_r(np.linspace(0.2, 0.8, len(top_features)))

        ax.barh(range(len(top_features)), top_features['Importance_Relative'],
                color=colors, edgecolor='black', linewidth=0.5)

        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(
            [f.replace('_', ' ').title() for f in top_features['Feature']],
            fontsize=9
        )
        ax.set_xlabel('Relative Importance (%)', fontweight='bold')
        ax.set_title(
            'Key Predictive Features for Equipment Failure\n(Nigerian O&G Facilities)',
            fontweight='bold', pad=15
        )

        for i, (idx, row) in enumerate(top_features.iterrows()):
            ax.text(row['Importance_Relative'] + 0.5, i,
                    f"{row['Importance_Relative']:.1f}%",
                    va='center', fontsize=8)

        ax.set_xlim(0, top_features['Importance_Relative'].max() * 1.15)
        ax.grid(axis='x', alpha=0.3)
        plt.tight_layout()

        for fmt in ['png', 'pdf']:
            path = os.path.join(FIGURES_PAPER_DIR, f'fig1_feature_importance.{fmt}')
            plt.savefig(path, dpi=600 if fmt == 'png' else 300,
                        bbox_inches='tight', facecolor='white')
        plt.close()
        print("Generated: fig1_feature_importance")

    def fig2_model_performance(self):
        """Figure 2: ROC, Precision-Recall, and Calibration curves"""
        set_paper_style()

        fig = plt.figure(figsize=(12, 4))
        gs  = GridSpec(1, 3, figure=fig, wspace=0.3)

        y_true  = self.predictions['y_true']
        y_proba = self.predictions['y_proba']

        # ROC Curve
        ax1 = fig.add_subplot(gs[0, 0])
        fpr, tpr, _ = roc_curve(y_true, y_proba)
        roc_auc = auc(fpr, tpr)   # FIX: auc() instead of np.trapz

        ax1.plot(fpr, tpr, 'b-', linewidth=2, label=f'AUC = {roc_auc:.3f}')
        ax1.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5)
        ax1.fill_between(fpr, tpr, alpha=0.2)
        ax1.set_xlabel('False Positive Rate', fontweight='bold')
        ax1.set_ylabel('True Positive Rate', fontweight='bold')
        ax1.set_title('(a) ROC Curve', fontweight='bold')
        ax1.legend(loc='lower right')
        ax1.grid(alpha=0.3)

        # Precision-Recall Curve
        ax2 = fig.add_subplot(gs[0, 1])
        precision, recall, _ = precision_recall_curve(y_true, y_proba)
        pr_auc = auc(recall, precision)   # FIX: auc() instead of np.trapz

        ax2.plot(recall, precision, 'r-', linewidth=2, label=f'AP = {pr_auc:.3f}')
        ax2.fill_between(recall, precision, alpha=0.2, color='red')
        ax2.set_xlabel('Recall', fontweight='bold')
        ax2.set_ylabel('Precision', fontweight='bold')
        ax2.set_title('(b) Precision-Recall Curve', fontweight='bold')
        ax2.legend(loc='lower left')
        ax2.grid(alpha=0.3)

        # Calibration Plot
        ax3 = fig.add_subplot(gs[0, 2])
        prob_true, prob_pred = self._calibration_data(y_true, y_proba)

        ax3.plot(prob_pred, prob_true, 's-', markersize=6, linewidth=2,
                 label='Model', color='green')
        ax3.plot([0, 1], [0, 1], 'k--', linewidth=1, alpha=0.5, label='Perfect')
        ax3.set_xlabel('Mean Predicted Probability', fontweight='bold')
        ax3.set_ylabel('Fraction of Positives', fontweight='bold')
        ax3.set_title('(c) Calibration Plot', fontweight='bold')
        ax3.legend()
        ax3.grid(alpha=0.3)

        plt.suptitle('Model Performance Metrics - XGBoost/LightGBM Ensemble',
                     fontweight='bold', fontsize=12, y=1.02)

        for fmt in ['png', 'pdf']:
            path = os.path.join(FIGURES_PAPER_DIR, f'fig2_model_performance.{fmt}')
            plt.savefig(path, dpi=600 if fmt == 'png' else 300,
                        bbox_inches='tight', facecolor='white')
        plt.close()
        print("Generated: fig2_model_performance")

    def _calibration_data(self, y_true, y_prob, n_bins=10):
        """Helper for calibration curve."""
        bin_boundaries = np.linspace(0, 1, n_bins + 1)
        bin_lowers     = bin_boundaries[:-1]
        bin_uppers     = bin_boundaries[1:]

        bin_centers    = []
        bin_accuracies = []

        for lower, upper in zip(bin_lowers, bin_uppers):
            in_bin       = (y_prob > lower) & (y_prob <= upper)
            prop_in_bin  = in_bin.mean()
            if prop_in_bin > 0:
                accuracy_in_bin = y_true[in_bin].mean()
                bin_centers.append((lower + upper) / 2)
                bin_accuracies.append(accuracy_in_bin)

        return np.array(bin_accuracies), np.array(bin_centers)

    def fig3_api581_risk_matrix(self):
        """Figure 3: API 581 Risk Matrix visualization"""
        set_paper_style()

        if self.risk_data is None:
            print("Risk data not available")
            return

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        risk_order  = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        risk_counts = self.risk_data['risk_tier_ml'].value_counts()
        risk_counts = risk_counts.reindex(risk_order).fillna(0)
        colors      = [RISK_COLORS[r] for r in risk_order]

        wedges, texts, autotexts = ax1.pie(
            risk_counts, labels=risk_order, autopct='%1.1f%%',
            colors=colors, startangle=90, textprops={'fontsize': 9}
        )
        centre_circle = plt.Circle((0, 0), 0.70, fc='white')
        ax1.add_artist(centre_circle)
        ax1.text(0, 0, f'Total\n{len(self.risk_data)}',
                 ha='center', va='center', fontsize=12, fontweight='bold')
        ax1.set_title('(a) Risk Tier Distribution\n(ML-Based Assessment)',
                      fontweight='bold')

        sample_data = self.risk_data.sample(min(1000, len(self.risk_data)))
        for tier in risk_order:
            tier_data = sample_data[sample_data['risk_tier_ml'] == tier]
            ax2.scatter(tier_data['ml_pof'], tier_data['cof_ngn'] / 1e6,
                        c=RISK_COLORS[tier], label=tier, alpha=0.6, s=30,
                        edgecolors='black', linewidth=0.3)

        ax2.set_xlabel('Probability of Failure (ML-Predicted)', fontweight='bold')
        ax2.set_ylabel('Consequence of Failure (Million NGN)', fontweight='bold')
        ax2.set_title('(b) API 581 Risk Matrix\n(POF vs COF)', fontweight='bold')
        ax2.legend(title='Risk Tier', loc='upper left', fontsize=8)
        ax2.grid(alpha=0.3)
        ax2.set_xlim(0, self.risk_data['ml_pof'].max() * 1.1)

        plt.tight_layout()

        for fmt in ['png', 'pdf']:
            path = os.path.join(FIGURES_PAPER_DIR, f'fig3_api581_risk_matrix.{fmt}')
            plt.savefig(path, dpi=600 if fmt == 'png' else 300,
                        bbox_inches='tight', facecolor='white')
        plt.close()
        print("Generated: fig3_api581_risk_matrix")

    def fig4_pof_distribution(self):
        """Figure 4: POF distribution by equipment type"""
        set_paper_style()

        if self.risk_data is None:
            print("Risk data not available")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        equipment_types = self.risk_data['equipment_type'].unique()
        pof_by_type = [
            self.risk_data[self.risk_data['equipment_type'] == et]['ml_pof'].values
            for et in equipment_types
        ]

        bp = ax.boxplot(pof_by_type,
                        labels=[et.replace(' ', '\n') for et in equipment_types],
                        patch_artist=True, notch=True)

        colors = plt.cm.Set3(np.linspace(0, 1, len(equipment_types)))
        for patch, color in zip(bp['boxes'], colors):
            patch.set_facecolor(color)
            patch.set_alpha(0.7)

        ax.set_ylabel('Probability of Failure (ML-Predicted)', fontweight='bold')
        ax.set_xlabel('Equipment Type', fontweight='bold')
        ax.set_title('POF Distribution by Equipment Type\n(Nigerian O&G Facilities)',
                     fontweight='bold', pad=15)
        ax.grid(axis='y', alpha=0.3)

        for i, et in enumerate(equipment_types):
            n = len(self.risk_data[self.risk_data['equipment_type'] == et])
            ax.text(i + 1, -0.005, f'n={n}', ha='center', fontsize=8,
                    transform=ax.get_xaxis_transform())

        plt.tight_layout()

        for fmt in ['png', 'pdf']:
            path = os.path.join(FIGURES_PAPER_DIR, f'fig4_pof_distribution.{fmt}')
            plt.savefig(path, dpi=600 if fmt == 'png' else 300,
                        bbox_inches='tight', facecolor='white')
        plt.close()
        print("Generated: fig4_pof_distribution")

    def fig5_economic_impact(self):
        """Figure 5: Economic impact comparison (Traditional vs ML-RBI)"""
        set_paper_style()

        econ_path = os.path.join(self.results_dir, 'economic_analysis.csv')
        if not os.path.exists(econ_path):
            print("Economic analysis not available")
            return

        econ = pd.read_csv(econ_path).iloc[0]

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

        # Cost Comparison
        categories      = ['Traditional\n(Time-Based)', 'ML-Enhanced\n(Risk-Based)']
        inspection_costs = [econ['annual_inspection_traditional'] / 1e6,
                            econ['annual_inspection_ml'] / 1e6]
        failure_costs   = [0, econ['prevented_failure_cost'] / 1e6 * 0.1]

        x     = np.arange(len(categories))
        width = 0.5

        ax1.bar(x, inspection_costs, width, label='Inspection Cost',
                color='#3498db', edgecolor='black')
        ax1.bar(x, failure_costs, width, bottom=inspection_costs,
                label='Failure Cost', color='#e74c3c', edgecolor='black')

        ax1.set_ylabel('Annual Cost (Million NGN)', fontweight='bold')
        ax1.set_title('(a) Total Cost of Integrity Management', fontweight='bold')
        ax1.set_xticks(x)
        ax1.set_xticklabels(categories)
        ax1.legend()
        ax1.grid(axis='y', alpha=0.3)

        for i, (ic, fc) in enumerate(zip(inspection_costs, failure_costs)):
            total = ic + fc
            ax1.text(i, total + 1, f'{total:.1f}M', ha='center', fontweight='bold')

        # Savings Breakdown
        savings_categories = ['Inspection\nSavings', 'Prevented\nFailures', 'Net\nBenefit']
        savings_values     = [
            econ['inspection_savings'] / 1e6,
            econ['prevented_failure_cost'] / 1e6,
            econ['net_annual_benefit'] / 1e6
        ]
        colors_savings = ['#2ecc71', '#27ae60', '#1e8449']

        bars = ax2.bar(savings_categories, savings_values, color=colors_savings,
                       edgecolor='black', linewidth=1)
        ax2.set_ylabel('Annual Value (Million NGN)', fontweight='bold')
        ax2.set_title('(b) ML-RBI Economic Benefits', fontweight='bold')
        ax2.grid(axis='y', alpha=0.3)

        for bar, val in zip(bars, savings_values):
            height = bar.get_height()
            ax2.text(bar.get_x() + bar.get_width() / 2., height + height * 0.01,
                     f'{val:.1f}M', ha='center', va='bottom', fontweight='bold')

        # FIX: pull roi value out before f-string to avoid backslash-in-fstring error
        roi_val  = econ['roi_percent']
        roi_text = f"ROI: {roi_val:.0f}%"
        ax2.text(0.95, 0.95, roi_text, transform=ax2.transAxes, fontsize=14,
                 fontweight='bold', ha='right', va='top',
                 bbox=dict(boxstyle='round', facecolor='yellow', alpha=0.7))

        plt.tight_layout()

        for fmt in ['png', 'pdf']:
            path = os.path.join(FIGURES_PAPER_DIR, f'fig5_economic_impact.{fmt}')
            plt.savefig(path, dpi=600 if fmt == 'png' else 300,
                        bbox_inches='tight', facecolor='white')
        plt.close()
        print("Generated: fig5_economic_impact")

    # =========================================================================
    # PRESENTATION FIGURES
    # =========================================================================

    def pres_01_overview(self):
        """Presentation slide: Framework overview diagram"""
        set_presentation_style()

        fig, ax = plt.subplots(figsize=(12, 8))
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 10)
        ax.axis('off')

        ax.text(5, 9.5, 'ML-Enhanced Risk-Based Inspection Framework',
                ha='center', fontsize=20, fontweight='bold')

        boxes = [
            (1,  7,   'Nigerian O&G\nFacility Data',          '#3498db'),
            (5,  7,   'Feature Engineering\n& API 581 Alignment', '#9b59b6'),
            (9,  7,   'XGBoost/LightGBM\nPOF Prediction',     '#e74c3c'),
            (3,  4,   'Risk Matrix\n(POF x COF)',              '#f39c12'),
            (7,  4,   'Optimized Inspection\nIntervals',       '#2ecc71'),
            (5,  1.5, '30-40% Cost Reduction\n60-70% Fewer Shutdowns', '#1abc9c')
        ]

        for x, y, text, color in boxes:
            rect = mpatches.FancyBboxPatch(
                (x - 0.8, y - 0.6), 1.6, 1.2,
                boxstyle="round,pad=0.1",
                facecolor=color, edgecolor='black', linewidth=2, alpha=0.8
            )
            ax.add_patch(rect)
            ax.text(x, y, text, ha='center', va='center',
                    fontsize=12, fontweight='bold', color='white')

        arrows = [
            (1.8, 7,   4.2, 7),
            (5.8, 7,   8.2, 7),
            (9,   6.4, 7,   4.6),
            (5,   6.4, 3,   4.6),
            (3.8, 4,   6.2, 4),
            (5,   3.4, 5,   2.1)
        ]
        for x1, y1, x2, y2 in arrows:
            ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                        arrowprops=dict(arrowstyle='->', lw=3, color='black'))

        plt.tight_layout()
        path = os.path.join(FIGURES_PRESENTATION_DIR, 'pres_01_overview.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("Generated: pres_01_overview")

    def pres_02_risk_distribution(self):
        """Presentation slide: Risk distribution with large fonts"""
        set_presentation_style()

        if self.risk_data is None:
            print("Risk data not available")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        risk_order  = ['Very Low', 'Low', 'Medium', 'High', 'Very High']
        risk_counts = self.risk_data['risk_tier_ml'].value_counts()
        risk_counts = risk_counts.reindex(risk_order).fillna(0)
        colors      = [RISK_COLORS_PRES[r] for r in risk_order]

        bars = ax.bar(risk_order, risk_counts, color=colors,
                      edgecolor='black', linewidth=2)

        ax.set_ylabel('Number of Equipment', fontweight='bold')
        ax.set_xlabel('Risk Tier (API 581)', fontweight='bold')
        ax.set_title('Equipment Risk Distribution - ML Enhanced RBI',
                     fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)

        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width() / 2., height,
                    f'{int(height)}', ha='center', va='bottom',
                    fontsize=14, fontweight='bold')

        plt.tight_layout()
        path = os.path.join(FIGURES_PRESENTATION_DIR, 'pres_02_risk_distribution.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("Generated: pres_02_risk_distribution")

    def pres_03_inspection_timeline(self):
        """Presentation slide: Inspection interval comparison"""
        set_presentation_style()

        if self.risk_data is None:
            print("Risk data not available")
            return

        fig, ax = plt.subplots(figsize=(12, 8))

        interval_comparison = self.risk_data.groupby('equipment_type').agg(
            traditional_interval_months=('traditional_interval_months', 'first'),
            recommended_interval_months=('recommended_interval_months', 'mean')
        ).reset_index()

        x     = np.arange(len(interval_comparison))
        width = 0.35

        ax.bar(x - width / 2, interval_comparison['traditional_interval_months'],
               width, label='Traditional (Time-Based)',
               color='#e74c3c', edgecolor='black', linewidth=2)
        ax.bar(x + width / 2, interval_comparison['recommended_interval_months'],
               width, label='ML-RBI (Optimized)',
               color='#2ecc71', edgecolor='black', linewidth=2)

        ax.set_ylabel('Inspection Interval (Months)', fontweight='bold')
        ax.set_xlabel('Equipment Type', fontweight='bold')
        ax.set_title('Inspection Interval Optimization', fontweight='bold', pad=20)
        ax.set_xticks(x)
        ax.set_xticklabels(
            [et.replace(' ', '\n') for et in interval_comparison['equipment_type']],
            rotation=0
        )
        ax.legend(fontsize=12)
        ax.grid(axis='y', alpha=0.3)

        plt.tight_layout()
        path = os.path.join(FIGURES_PRESENTATION_DIR, 'pres_03_inspection_timeline.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("Generated: pres_03_inspection_timeline")

    def pres_04_roi(self):
        """Presentation slide: ROI and savings"""
        set_presentation_style()

        econ_path = os.path.join(self.results_dir, 'economic_analysis.csv')
        if not os.path.exists(econ_path):
            print("Economic analysis not available")
            return

        econ = pd.read_csv(econ_path).iloc[0]

        fig, ax = plt.subplots(figsize=(12, 8))

        # FIX: pull roi_percent out first — no backslash needed in f-string
        roi_val = min(econ['roi_percent'], 500)
        roi_full = econ['roi_percent']

        metrics = ['Cost\nReduction', 'Shutdown\nReduction', 'ROI']
        values  = [
            econ['inspection_reduction_percent'],
            econ['shutdown_reduction_percent'],
            roi_val
        ]
        colors = ['#3498db', '#9b59b6', '#e74c3c']

        bars = ax.bar(metrics, values, color=colors, edgecolor='black', linewidth=2)

        ax.set_ylabel('Percentage (%)', fontweight='bold')
        ax.set_title('ML-RBI Performance Metrics', fontweight='bold', pad=20)
        ax.grid(axis='y', alpha=0.3)
        ax.set_ylim(0, max(values) * 1.2)

        for bar, val, is_roi in zip(bars, values, [False, False, True]):
            height = bar.get_height()
            # FIX: no backslash in f-string — use pre-extracted variable
            label  = f'{roi_full:.0f}%' if is_roi else f'{val:.0f}%'
            ax.text(bar.get_x() + bar.get_width() / 2., height + 5,
                    label, ha='center', va='bottom',
                    fontsize=16, fontweight='bold')

        plt.tight_layout()
        path = os.path.join(FIGURES_PRESENTATION_DIR, 'pres_04_roi.png')
        plt.savefig(path, dpi=300, bbox_inches='tight', facecolor='white')
        plt.close()
        print("Generated: pres_04_roi")

    def generate_all(self):
        """Generate all paper and presentation figures."""
        print("\n" + "=" * 80)
        print("GENERATING PUBLICATION FIGURES")
        print("=" * 80)

        print("\n--- Paper Figures (High Resolution) ---")
        self.fig1_feature_importance()
        self.fig2_model_performance()
        self.fig3_api581_risk_matrix()
        self.fig4_pof_distribution()
        self.fig5_economic_impact()

        print("\n--- Presentation Figures (Large Format) ---")
        self.pres_01_overview()
        self.pres_02_risk_distribution()
        self.pres_03_inspection_timeline()
        self.pres_04_roi()

        print("\n" + "=" * 80)
        print("ALL FIGURES GENERATED SUCCESSFULLY")
        print(f"Paper figures:        {FIGURES_PAPER_DIR}")
        print(f"Presentation figures: {FIGURES_PRESENTATION_DIR}")
        print("=" * 80)


if __name__ == "__main__":
    generator = FigureGenerator()
    generator.generate_all()