# ml-risk-inspection-og

ML-Enhanced Risk-Based Inspection (RBI) for Nigerian Oil & Gas.

This project replaces generic inspection tables with machine learning models trained on local degradation data to improve inspection planning, cut costs, and reduce unplanned shutdowns.

## What it does

- Uses XGBoost and LightGBM for inspection risk prediction.
- Trains on Nigerian oil & gas degradation and operational data.
- Supports smarter inspection prioritization for assets with varying risk levels.
- Helps replace static RBI schedules with adaptive, data-driven decisions.

## Why it matters

Traditional RBI approaches often rely on generic inspection tables that do not fully reflect local operating conditions. This project introduces a more realistic, site-aware framework for Nigerian oil & gas assets.

Expected impact:
- 30–40% cost reduction.
- 60–70% fewer unplanned shutdowns.
- Better targeting of high-risk equipment.
- Improved maintenance planning and resource allocation.

## Validation

The framework was validated using 20,000+ field records, with an AUC-ROC of 0.858, showing strong predictive performance for risk-based inspection use cases.

## Repository structure

```bash
ml-risk-inspection-og/
├── data/
├── figures/
├── results/
├── api581_framework.py
├── config.py
├── data_pipeline.py
├── generate_figures.py
├── model_training.py
└── rbi_framework.py
```

## Authors

- Emmanuel Alao
- Maduabuchi David

## Tech stack

- Python
- XGBoost
- LightGBM
- pandas
- scikit-learn
- matplotlib / seaborn

## Notes

This repository is part of an ML-driven RBI workflow for improving inspection decision-making in oil and gas asset management.

## License

MIT LICENSE.
