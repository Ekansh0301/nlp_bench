# Hindi Word Order Classification Analysis Results

Generated on: 2025-06-28 21:24:59

## Directory Structure

```
hindi_word_order_analysis_nonlinfinal/
├── README.md (this file)
├── plots/
│   ├── enhanced_feature_distributions.png
│   ├── enhanced_feature_correlations.png
│   ├── enhanced_feature_coefficients.png
│   ├── enhanced_calibration_analysis.png
│   ├── partial_dependence_plots.png
│   ├── nonlinear_effects_binned.png
│   ├── decision_tree_boundaries.png
│   └── [other plots]
├── results/
│   ├── comprehensive_analysis_report.txt
│   ├── feature_statistics.csv
│   ├── coefficient_analysis.csv
│   ├── permutation_importance.csv
│   ├── enhanced_error_analysis.csv
│   └── nonlinear_analysis_results.txt
├── models/
│   ├── final_logistic_model.pkl
│   └── feature_scaler.pkl
└── [data files]
```

## Key Results

- **Dominant Feature**: Trigram Surprisal
- **Model Calibration**: Brier Score = 0.0776
- **Non-linearities**: Random Forest outperforms by 0.86%
