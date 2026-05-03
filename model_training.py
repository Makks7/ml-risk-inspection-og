"""
NAICE-SPE Paper: Advanced Model Training Pipeline
XGBoost & LightGBM for Risk-Based Inspection Prediction
Optimized for Nigerian O&G Mechanical Integrity Data
"""

import os
import numpy as np
import pandas as pd
import joblib
import warnings
from datetime import datetime
from typing import Dict, Tuple, List, Any

# ML Libraries
import xgboost as xgb
import lightgbm as lgb
from sklearn.model_selection import StratifiedKFold, RandomizedSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    roc_auc_score, average_precision_score, f1_score,
    precision_recall_curve, roc_curve, confusion_matrix,
    classification_report, matthews_corrcoef, brier_score_loss,
    auc  # FIX: explicitly imported — replaces np.trapz removed in NumPy 2.0
)
from sklearn.calibration import CalibratedClassifierCV, calibration_curve

# Config import
from config import RESULTS_DIR, RISK_COLORS

warnings.filterwarnings('ignore')

# Set random seed for reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)


class RBIModelTrainer:
    """
    Advanced ML pipeline for Risk-Based Inspection modeling.
    Implements XGBoost and LightGBM with hyperparameter optimization,
    temporal validation, and model calibration for probability of failure prediction.
    """

    def __init__(self, X: pd.DataFrame, y: pd.Series, feature_names: List[str]):
        self.X = X
        self.y = y
        self.feature_names = feature_names
        self.models = {}
        self.results = {}
        self.best_model = None
        self.best_model_name = None

        # Temporal split indices (70-15-15)
        n_samples = len(X)
        self.train_idx = slice(0, int(n_samples * 0.7))
        self.val_idx = slice(int(n_samples * 0.7), int(n_samples * 0.85))
        self.test_idx = slice(int(n_samples * 0.85), n_samples)

        print("=" * 80)
        print("NAICE-SPE: Advanced ML Training Pipeline")
        print("=" * 80)
        print(f"Total samples: {n_samples}")
        print(f"Training: {self.train_idx.stop - self.train_idx.start} | "
              f"Validation: {self.val_idx.stop - self.val_idx.start} | "
              f"Test: {self.test_idx.stop - self.test_idx.start}")
        print(f"Features: {len(feature_names)}")
        print(f"Class distribution: {dict(y.value_counts().sort_index())}")

    def _get_splits(self) -> Tuple:
        """Return train/val/test splits maintaining temporal order."""
        X_train = self.X.iloc[self.train_idx]
        X_val   = self.X.iloc[self.val_idx]
        X_test  = self.X.iloc[self.test_idx]
        y_train = self.y.iloc[self.train_idx]
        y_val   = self.y.iloc[self.val_idx]
        y_test  = self.y.iloc[self.test_idx]
        return X_train, X_val, X_test, y_train, y_val, y_test

    def optimize_xgboost(self, n_iter: int = 50) -> Dict:
        """
        Optimize XGBoost hyperparameters using RandomizedSearchCV.
        Focus on imbalanced dataset handling and Nigerian O&G data characteristics.
        """
        print("\n" + "-" * 60)
        print("Optimizing XGBoost Hyperparameters")
        print("-" * 60)

        X_train, X_val, X_test, y_train, y_val, y_test = self._get_splits()

        # Calculate scale_pos_weight for imbalanced data
        scale_pos_weight = len(y_train[y_train == 0]) / len(y_train[y_train == 1])

        param_distributions = {
            'max_depth':        [4, 6, 8, 10, 12],
            'learning_rate':    [0.01, 0.05, 0.1, 0.15],
            'n_estimators':     [200, 300, 500, 700],
            'min_child_weight': [1, 3, 5, 7],
            'subsample':        [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree': [0.6, 0.7, 0.8, 0.9],
            'gamma':            [0, 0.1, 0.2, 0.3],
            'reg_alpha':        [0, 0.1, 0.5, 1],
            'reg_lambda':       [1, 1.5, 2, 3]
        }

        base_model = xgb.XGBClassifier(
            objective='binary:logistic',
            eval_metric='aucpr',
            scale_pos_weight=scale_pos_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1
            # NOTE: use_label_encoder removed — deprecated in XGBoost >= 1.6
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=n_iter,
            scoring='average_precision',
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)

        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV Average Precision: {search.best_score_:.4f}")

        best_xgb = search.best_estimator_
        val_proba = best_xgb.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        val_ap  = average_precision_score(y_val, val_proba)

        print(f"Validation AUC-ROC: {val_auc:.4f}")
        print(f"Validation Average Precision: {val_ap:.4f}")

        self.models['xgboost'] = best_xgb
        return {
            'best_params': search.best_params_,
            'cv_score':    search.best_score_,
            'val_auc':     val_auc,
            'val_ap':      val_ap,
            'model':       best_xgb
        }

    def optimize_lightgbm(self, n_iter: int = 50) -> Dict:
        """
        Optimize LightGBM hyperparameters.
        Optimized for speed and performance on tabular data.
        """
        print("\n" + "-" * 60)
        print("Optimizing LightGBM Hyperparameters")
        print("-" * 60)

        # FIX: was 'y_test' in 4th position — corrected to 'y_train'
        X_train, X_val, X_test, y_train, y_val, y_test = self._get_splits()

        # Calculate class weight
        class_weight = {0: 1, 1: len(self.y[self.y == 0]) / len(self.y[self.y == 1])}

        param_distributions = {
            'num_leaves':        [31, 50, 70, 100, 150],
            'max_depth':         [6, 8, 10, 12, -1],
            'learning_rate':     [0.01, 0.05, 0.1, 0.15],
            'n_estimators':      [200, 300, 500, 700],
            'min_child_samples': [10, 20, 30, 50],
            'subsample':         [0.6, 0.7, 0.8, 0.9],
            'colsample_bytree':  [0.6, 0.7, 0.8, 0.9],
            'reg_alpha':         [0, 0.1, 0.5, 1],
            'reg_lambda':        [0, 0.1, 0.5, 1],
            'min_split_gain':    [0, 0.01, 0.1]
        }

        base_model = lgb.LGBMClassifier(
            objective='binary',
            metric='average_precision',
            class_weight=class_weight,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=-1  # FIX: 'verbose' renamed to 'verbosity' in newer LightGBM
        )

        cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

        search = RandomizedSearchCV(
            base_model,
            param_distributions,
            n_iter=n_iter,
            scoring='average_precision',
            cv=cv,
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbose=1
        )

        search.fit(X_train, y_train)

        print(f"Best parameters: {search.best_params_}")
        print(f"Best CV Average Precision: {search.best_score_:.4f}")

        best_lgb = search.best_estimator_
        val_proba = best_lgb.predict_proba(X_val)[:, 1]
        val_auc = roc_auc_score(y_val, val_proba)
        val_ap  = average_precision_score(y_val, val_proba)

        print(f"Validation AUC-ROC: {val_auc:.4f}")
        print(f"Validation Average Precision: {val_ap:.4f}")

        self.models['lightgbm'] = best_lgb
        return {
            'best_params': search.best_params_,
            'cv_score':    search.best_score_,
            'val_auc':     val_auc,
            'val_ap':      val_ap,
            'model':       best_lgb
        }

    def calibrate_probabilities(self, model_name: str) -> Any:
        """
        Apply Platt scaling (sigmoid calibration) to improve probability estimates.
        Critical for risk-based decision making.
        """
        print(f"\nCalibrating {model_name} probabilities...")

        X_train, X_val, X_test, y_train, y_val, y_test = self._get_splits()
        base_model = self.models[model_name]

        calibrated = CalibratedClassifierCV(
            base_model,
            method='sigmoid',
            cv='prefit'
        )
        calibrated.fit(X_val, y_val)

        prob_pos = calibrated.predict_proba(X_test)[:, 1]
        brier = brier_score_loss(y_test, prob_pos)
        print(f"Brier score (calibrated): {brier:.4f}")

        return calibrated

    def evaluate_models(self) -> pd.DataFrame:
        """
        Comprehensive evaluation of all trained models on test set.
        Includes discrimination metrics, calibration, and business metrics.
        """
        print("\n" + "=" * 80)
        print("COMPREHENSIVE MODEL EVALUATION")
        print("=" * 80)

        X_train, X_val, X_test, y_train, y_val, y_test = self._get_splits()
        results = []

        for name, model in self.models.items():
            print(f"\nEvaluating {name.upper()}...")

            y_proba = model.predict_proba(X_test)[:, 1]
            y_pred  = model.predict(X_test)

            auc_roc       = roc_auc_score(y_test, y_proba)
            avg_precision = average_precision_score(y_test, y_proba)
            f1            = f1_score(y_test, y_pred)
            mcc           = matthews_corrcoef(y_test, y_pred)
            brier         = brier_score_loss(y_test, y_proba)

            precision, recall, _ = precision_recall_curve(y_test, y_proba)
            pr_auc = auc(recall, precision)  # FIX: np.trapz removed in NumPy 2.0

            result = {
                'Model':              name,
                'AUC-ROC':            auc_roc,
                'Average Precision':  avg_precision,
                'PR-AUC':             pr_auc,
                'F1-Score':           f1,
                'MCC':                mcc,
                'Brier Score':        brier,
                'Predictions':        y_proba,
                'Binary Predictions': y_pred
            }
            results.append(result)

            print(f"  AUC-ROC: {auc_roc:.4f}")
            print(f"  Average Precision: {avg_precision:.4f}")
            print(f"  PR-AUC: {pr_auc:.4f}")
            print(f"  F1-Score: {f1:.4f}")
            print(f"  MCC: {mcc:.4f}")
            print(f"  Brier Score: {brier:.4f}")

        # Select best model based on Average Precision (robust for imbalanced data)
        best_idx = np.argmax([r['Average Precision'] for r in results])
        self.best_model_name = results[best_idx]['Model']
        self.best_model      = self.models[self.best_model_name]

        print(f"\n{'='*80}")
        print(f"BEST MODEL: {self.best_model_name.upper()}")
        print(f"{'='*80}")

        self.test_predictions = results[best_idx]['Predictions']
        self.test_binary      = results[best_idx]['Binary Predictions']
        self.y_test           = y_test

        return pd.DataFrame([{k: v for k, v in r.items()
                               if k not in ['Predictions', 'Binary Predictions']}
                              for r in results])

    def get_feature_importance(self) -> pd.DataFrame:
        """Extract and rank feature importance from best model."""
        print("\nExtracting Feature Importance...")

        importance      = self.best_model.feature_importances_
        importance_type = 'Gain' if self.best_model_name == 'xgboost' else 'Split'

        importance_df = pd.DataFrame({
            'Feature':             self.feature_names,
            'Importance':          importance,
            'Importance_Relative': importance / importance.sum() * 100
        }).sort_values('Importance', ascending=False)

        print(f"\nTop 10 Most Important Features ({self.best_model_name}, {importance_type}):")
        for idx, row in importance_df.head(10).iterrows():
            print(f"  {row['Feature']:<30} {row['Importance_Relative']:.2f}%")

        return importance_df

    def save_models(self):
        """Save all models and results."""
        print("\n" + "-" * 60)
        print("Saving Models and Results")
        print("-" * 60)

        for name, model in self.models.items():
            path = os.path.join(RESULTS_DIR, f'model_{name}.pkl')
            joblib.dump(model, path)
            print(f"Saved: {path}")

        best_path = os.path.join(RESULTS_DIR, 'best_model.pkl')
        joblib.dump(self.best_model, best_path)
        print(f"Saved best model: {best_path}")

        pred_df = pd.DataFrame({
            'y_true':  self.y_test.values,
            'y_proba': self.test_predictions,
            'y_pred':  self.test_binary
        })
        pred_path = os.path.join(RESULTS_DIR, 'predictions.csv')
        pred_df.to_csv(pred_path, index=False)
        print(f"Saved predictions: {pred_path}")

        importance_df = self.get_feature_importance()
        imp_path = os.path.join(RESULTS_DIR, 'feature_importance.csv')
        importance_df.to_csv(imp_path, index=False)
        print(f"Saved feature importance: {imp_path}")


def train_and_evaluate(X: pd.DataFrame, y: pd.Series, feature_names: List[str]) -> RBIModelTrainer:
    """
    Main training pipeline execution.

    Returns:
    --------
    trainer : RBIModelTrainer
        Trained model container with all results
    """
    trainer = RBIModelTrainer(X, y, feature_names)

    xgb_results = trainer.optimize_xgboost(n_iter=30)
    lgb_results = trainer.optimize_lightgbm(n_iter=30)

    comparison_df = trainer.evaluate_models()
    comp_path = os.path.join(RESULTS_DIR, 'model_comparison.csv')
    comparison_df.to_csv(comp_path, index=False)
    print(f"\nSaved model comparison: {comp_path}")

    trainer.save_models()

    return trainer


if __name__ == "__main__":
    from data_pipeline import load_and_engineer_data, prepare_ml_features

    data_path = os.path.join('data', 'nigerian_og_rbi_dataset_2020_2025_v3.csv')
    df = load_and_engineer_data(data_path)
    X, y, features = prepare_ml_features(df)

    trainer = train_and_evaluate(X, y, features)