================================================================================
HINDI WORD ORDER CLASSIFICATION ANALYSIS - FILE INDEX
================================================================================

DIRECTORY STRUCTURE:
------------------------------
hindi_word_order_analysis/
├── README.txt (this file)
├── plots/
│   ├── feature_distributions.png
│   ├── feature_correlations.png
│   ├── feature_coefficients.png
│   ├── permutation_importance.png
│   ├── interaction_effects.png
│   ├── learning_curves.png
│   ├── model_confidence.png
│   ├── advanced_correlation_analysis.png
│   ├── roc_pr_curves.png
│   ├── coefficients_with_ci.png
│   ├── feature_distributions_comparison.png
│   ├── model_calibration_detailed.png
│   ├── bootstrap_analysis.png
│   └── feature_stability.png
└── results/
    ├── 01_basic_model_results.txt
    ├── 02_feature_analysis.txt
    ├── 03_advanced_statistical_analysis.txt
    ├── 04_error_analysis.txt
    └── 05_comprehensive_summary_report.txt

FILE DESCRIPTIONS:
------------------------------

PLOTS:
• feature_distributions.png - Distribution of each feature with statistics
• feature_correlations.png - Correlation matrix heatmap
• feature_coefficients.png - Logistic regression coefficients
• permutation_importance.png - Feature importance from permutation
• interaction_effects.png - Interaction effects visualization
• learning_curves.png - Model performance vs training size
• model_confidence.png - Prediction confidence analysis
• advanced_correlation_analysis.png - Hierarchical clustering of features
• roc_pr_curves.png - ROC and Precision-Recall curves
• coefficients_with_ci.png - Coefficients with confidence intervals
• feature_distributions_comparison.png - Reference vs Variant distributions
• model_calibration_detailed.png - Model calibration analysis
• bootstrap_analysis.png - Bootstrap confidence intervals
• feature_stability.png - Feature stability across subsamples

RESULTS:
• 01_basic_model_results.txt - Cross-validation and basic performance metrics
• 02_feature_analysis.txt - Feature correlations, coefficients, and importance
• 03_advanced_statistical_analysis.txt - Comprehensive statistical analysis
• 04_error_analysis.txt - Error patterns and analysis
• 05_comprehensive_summary_report.txt - Complete analysis summary

ANALYSIS OVERVIEW:
------------------------------
This analysis investigates Hindi word order preferences using machine learning
to distinguish reference sentences from grammatical variants. The study examines
the relative impact of:
- Trigram surprisal (lexical predictability)
- Dependency length (memory constraints)
- Information status (discourse factors)
- Positional language model scores
- Case marker transitions

KEY FINDINGS:
- Trigram surprisal is the strongest predictor (91%+ accuracy)
- Dependency length shows weak but significant effects
- Feature interactions provide modest improvements
- Model demonstrates excellent calibration and stability
