#!/usr/bin/env python3
"""
Balanced Dataset Logistic Regression for Hindi Sentence Classification
Creates balanced dataset with all reference sentences + 1 random variant per reference
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression, LogisticRegressionCV
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import (accuracy_score, classification_report, f1_score, 
                           precision_score, recall_score, confusion_matrix, 
                           roc_curve, auc, roc_auc_score, average_precision_score,
                           precision_recall_curve)
from sklearn.utils import resample
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
import os
from datetime import datetime
from collections import defaultdict
warnings.filterwarnings('ignore')

# Create output directory
output_dir = f'balanced_lr_classification_1'
os.makedirs(output_dir, exist_ok=True)
os.makedirs(f'{output_dir}/plots', exist_ok=True)

# Set plot style
plt.style.use('seaborn-v0_8-darkgrid')
sns.set_palette("husl")

print("="*100)
print("BALANCED DATASET LOGISTIC REGRESSION - HINDI SENTENCE CLASSIFICATION")
print("Creating balanced dataset: All references + 1 random variant per reference")
print(f"Report generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("="*100)

# Load all features
print("\n1. DATA LOADING")
print("-"*50)
print("Loading feature files...")

# Load sentences to understand structure
try:
    sentences_df = pd.read_csv('hutb-sentences.csv')
    print(f"Loaded {len(sentences_df)} sentences")
    print(f"Sample IDs: {sentences_df['Sentence ID'].head().tolist()}")
except:
    print("Note: hutb-sentences.csv not found, continuing with feature files only")

# Load features
surprisal_df = pd.read_csv('hutb_trigram_surprisals.csv')
surprisal_lookup = {row['sentence_id']: row['trigram_surprisal'] 
                   for _, row in surprisal_df.iterrows()}

dep_df = pd.read_csv('enhanced_dependency_detailed.csv')
dep_lookup = {row['sentence_id']: row['dependency_length'] 
              for _, row in dep_df.iterrows() if pd.notna(row['dependency_length'])}

is_df = pd.read_csv('hutb_is_scores.csv')
is_lookup = {row['sentence_id']: row['is_score'] 
             for _, row in is_df.iterrows()}

plm_df = pd.read_csv('hutb_plm_scores_robust.csv')
plm_df['positional_lm_score'] = -plm_df['positional_lm_score']
plm_lookup = {row['sentence_id']: row['positional_lm_score'] 
              for _, row in plm_df.iterrows()}

cm_df = pd.read_csv('hutb_case_marker_scores.csv')
cm_lookup = {row['sentence_id']: row['case_marker_score'] 
             for _, row in cm_df.iterrows()}

print(f"\nFeature coverage:")
print(f"  Surprisal: {len(surprisal_lookup)} sentences")
print(f"  Dependency: {len(dep_lookup)} sentences")
print(f"  Information Status: {len(is_lookup)} sentences")
print(f"  Positional LM: {len(plm_lookup)} sentences")
print(f"  Case Marker: {len(cm_lookup)} sentences")

# Create dataset structure to identify sentence families
print("\n2. IDENTIFYING SENTENCE FAMILIES")
print("-"*50)

sentence_families = defaultdict(list)
all_sentence_ids = set(surprisal_lookup.keys())

for sent_id in all_sentence_ids:
    if '.' in sent_id:
        # Extract base ID (everything before the last dot)
        base_id = sent_id.rsplit('.', 1)[0]
        variant_num = int(sent_id.rsplit('.', 1)[1])
        sentence_families[base_id].append((sent_id, variant_num))

# Sort variants within each family
for base_id in sentence_families:
    sentence_families[base_id].sort(key=lambda x: x[1])

print(f"Found {len(sentence_families)} sentence families")
print(f"Average variants per family: {np.mean([len(v) for v in sentence_families.values()]):.2f}")

# Verify reference sentences exist
missing_refs = []
for base_id, variants in sentence_families.items():
    variant_nums = [v[1] for v in variants]
    if 0 not in variant_nums:
        missing_refs.append(base_id)

if missing_refs:
    print(f"WARNING: {len(missing_refs)} families missing reference sentence (.0)")
    print(f"Examples: {missing_refs[:5]}")

# Create balanced dataset
print("\n3. CREATING BALANCED DATASET")
print("-"*50)

np.random.seed(42)  # For reproducibility
balanced_data = []

families_with_complete_data = 0
families_skipped = 0

for base_id, variants in tqdm(sentence_families.items(), desc="Processing families"):
    # Find reference sentence
    ref_sent_id = None
    variant_sent_ids = []
    
    for sent_id, variant_num in variants:
        if variant_num == 0:
            ref_sent_id = sent_id
        else:
            variant_sent_ids.append(sent_id)
    
    # Skip if no reference sentence
    if ref_sent_id is None:
        families_skipped += 1
        continue
    
    # Skip if no variants
    if len(variant_sent_ids) == 0:
        families_skipped += 1
        continue
    
    # Check if reference has all features
    if not all([
        ref_sent_id in surprisal_lookup,
        ref_sent_id in dep_lookup,
        ref_sent_id in plm_lookup,
        ref_sent_id in cm_lookup
    ]):
        families_skipped += 1
        continue
    
    # Find variants with complete features
    valid_variants = []
    for var_id in variant_sent_ids:
        if all([
            var_id in surprisal_lookup,
            var_id in dep_lookup,
            var_id in plm_lookup,
            var_id in cm_lookup
        ]):
            valid_variants.append(var_id)
    
    # Skip if no valid variants
    if len(valid_variants) == 0:
        families_skipped += 1
        continue
    
    # Add reference sentence (label = 1)
    balanced_data.append([
        surprisal_lookup[ref_sent_id],
        dep_lookup[ref_sent_id],
        is_lookup.get(ref_sent_id, 0),
        plm_lookup[ref_sent_id],
        cm_lookup[ref_sent_id],
        1,  # is_reference
        ref_sent_id,
        base_id  # family ID
    ])
    
    # Randomly select one variant (label = 0)
    selected_variant = np.random.choice(valid_variants)
    balanced_data.append([
        surprisal_lookup[selected_variant],
        dep_lookup[selected_variant],
        is_lookup.get(selected_variant, 0),
        plm_lookup[selected_variant],
        cm_lookup[selected_variant],
        0,  # is_reference
        selected_variant,
        base_id  # family ID
    ])
    
    families_with_complete_data += 1

print(f"\nDataset creation complete:")
print(f"  Families with complete data: {families_with_complete_data}")
print(f"  Families skipped: {families_skipped}")
print(f"  Total samples: {len(balanced_data)}")
print(f"  References: {sum(1 for x in balanced_data if x[5] == 1)}")
print(f"  Variants: {sum(1 for x in balanced_data if x[5] == 0)}")

# Convert to arrays
data = np.array(balanced_data, dtype=object)
X = np.array([row[:5] for row in data], dtype=float)
y = np.array([row[5] for row in data], dtype=int)
sentence_ids = [row[6] for row in data]
family_ids = [row[7] for row in data]

feature_names = ['Trigram Surprisal', 'Dependency Length', 'Information Status',
                 'Positional LM', 'Case Marker Score']

# Verify balance
print(f"\nClass distribution verification:")
print(f"  Class 0 (variants): {np.sum(y == 0)} ({np.mean(y == 0)*100:.1f}%)")
print(f"  Class 1 (references): {np.sum(y == 1)} ({np.mean(y == 1)*100:.1f}%)")

# Family-aware train-test split
print("\n4. FAMILY-AWARE TRAIN-TEST SPLIT")
print("-"*50)

def family_aware_split(family_ids, test_size=0.2, val_size=0.1, random_state=42):
    """Split data ensuring families stay together"""
    np.random.seed(random_state)
    
    # Get unique families
    unique_families = list(set(family_ids))
    np.random.shuffle(unique_families)
    
    # Calculate split sizes
    n_families = len(unique_families)
    n_test = int(n_families * test_size)
    n_val = int(n_families * val_size)
    
    # Split families
    test_families = set(unique_families[:n_test])
    val_families = set(unique_families[n_test:n_test + n_val])
    train_families = set(unique_families[n_test + n_val:])
    
    # Get indices
    train_idx = [i for i, fam in enumerate(family_ids) if fam in train_families]
    val_idx = [i for i, fam in enumerate(family_ids) if fam in val_families]
    test_idx = [i for i, fam in enumerate(family_ids) if fam in test_families]
    
    return train_idx, val_idx, test_idx

train_idx, val_idx, test_idx = family_aware_split(family_ids, test_size=0.2, val_size=0.1)

# Scale features
scaler = RobustScaler()
X_scaled = scaler.fit_transform(X)

# Split data
X_train = X_scaled[train_idx]
X_val = X_scaled[val_idx]
X_test = X_scaled[test_idx]
y_train = y[train_idx]
y_val = y[val_idx]
y_test = y[test_idx]

print(f"Dataset splits:")
print(f"  Train: {len(X_train)} samples ({np.mean(y_train)*100:.1f}% references)")
print(f"  Val: {len(X_val)} samples ({np.mean(y_val)*100:.1f}% references)")
print(f"  Test: {len(X_test)} samples ({np.mean(y_test)*100:.1f}% references)")

# Feature analysis
print("\n5. FEATURE ANALYSIS")
print("-"*50)

# Feature statistics
feature_stats = pd.DataFrame(X, columns=feature_names).describe()
print("\nFeature statistics (before scaling):")
print(feature_stats)

# Feature correlations
correlation_matrix = pd.DataFrame(X, columns=feature_names).corr()
plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, fmt='.3f', cmap='coolwarm', center=0,
            square=True, linewidths=1)
plt.title('Feature Correlation Matrix', fontsize=16, fontweight='bold')
plt.tight_layout()
plt.savefig(f'{output_dir}/plots/feature_correlations.png', dpi=300, bbox_inches='tight')
plt.close()

# Feature distributions by class
fig, axes = plt.subplots(2, 3, figsize=(15, 10))
axes = axes.ravel()

for idx, feature in enumerate(feature_names):
    ax = axes[idx]
    
    # Plot distributions
    ref_values = X[y == 1, idx]
    var_values = X[y == 0, idx]
    
    ax.hist(ref_values, bins=30, alpha=0.7, label='References', density=True, color='blue')
    ax.hist(var_values, bins=30, alpha=0.7, label='Variants', density=True, color='red')
    
    ax.set_xlabel(feature)
    ax.set_ylabel('Density')
    ax.set_title(f'{feature} Distribution by Class')
    ax.legend()
    
    # Add statistics
    ref_mean = np.mean(ref_values)
    var_mean = np.mean(var_values)
    ax.axvline(ref_mean, color='blue', linestyle='--', alpha=0.8)
    ax.axvline(var_mean, color='red', linestyle='--', alpha=0.8)

# Remove empty subplot
if len(feature_names) < 6:
    fig.delaxes(axes[-1])

plt.tight_layout()
plt.savefig(f'{output_dir}/plots/feature_distributions.png', dpi=300, bbox_inches='tight')
plt.close()

# Logistic Regression with hyperparameter tuning
print("\n6. LOGISTIC REGRESSION TRAINING")
print("-"*50)

# Define regularization parameters to test
C_values = [0.001, 0.01, 0.1, 1, 10, 100]
solvers = ['liblinear', 'lbfgs']
penalties = {
    'liblinear': ['l1', 'l2'],
    'lbfgs': ['l2']
}

# Grid search for best parameters
best_score = 0
best_params = {}
results = []

print("Performing hyperparameter search...")
for solver in solvers:
    for penalty in penalties[solver]:
        for C in C_values:
            # Use stratified k-fold cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            
            lr = LogisticRegression(
                C=C,
                solver=solver,
                penalty=penalty,
                max_iter=1000,
                random_state=42
            )
            
            # Cross-validation on training set
            cv_scores = cross_val_score(lr, X_train, y_train, cv=cv, scoring='roc_auc')
            mean_score = np.mean(cv_scores)
            std_score = np.std(cv_scores)
            
            results.append({
                'solver': solver,
                'penalty': penalty,
                'C': C,
                'mean_auc': mean_score,
                'std_auc': std_score
            })
            
            print(f"  {solver}-{penalty}, C={C}: AUC={mean_score:.4f} (±{std_score:.4f})")
            
            if mean_score > best_score:
                best_score = mean_score
                best_params = {
                    'solver': solver,
                    'penalty': penalty,
                    'C': C
                }

print(f"\nBest parameters: {best_params}")
print(f"Best CV AUC: {best_score:.4f}")

# Train final model with best parameters
print("\n7. FINAL MODEL TRAINING")
print("-"*50)

final_lr = LogisticRegression(
    C=best_params['C'],
    solver=best_params['solver'],
    penalty=best_params['penalty'],
    max_iter=1000,
    random_state=42
)

# Train on combined train+val set for final model
X_train_full = np.vstack([X_train, X_val])
y_train_full = np.hstack([y_train, y_val])

final_lr.fit(X_train_full, y_train_full)

# Evaluate on test set
y_pred = final_lr.predict(X_test)
y_proba = final_lr.predict_proba(X_test)[:, 1]

# Calculate metrics
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
auc_roc = roc_auc_score(y_test, y_proba)
pr_auc = average_precision_score(y_test, y_proba)

print("Test Set Performance:")
print(f"  Accuracy: {accuracy:.4f}")
print(f"  Precision: {precision:.4f}")
print(f"  Recall: {recall:.4f}")
print(f"  F1-Score: {f1:.4f}")
print(f"  AUC-ROC: {auc_roc:.4f}")
print(f"  PR-AUC: {pr_auc:.4f}")

# Feature importance analysis
print("\n8. FEATURE IMPORTANCE")
print("-"*50)

# Get coefficients
coefficients = final_lr.coef_[0]
feature_importance = pd.DataFrame({
    'Feature': feature_names,
    'Coefficient': coefficients,
    'Abs_Coefficient': np.abs(coefficients)
}).sort_values('Abs_Coefficient', ascending=False)

print("\nFeature Importance (by coefficient magnitude):")
print(feature_importance.to_string(index=False))

# Plot feature importance
plt.figure(figsize=(10, 6))
colors = ['green' if c > 0 else 'red' for c in feature_importance['Coefficient']]
plt.barh(feature_importance['Feature'], feature_importance['Coefficient'], color=colors)
plt.xlabel('Coefficient Value', fontsize=12)
plt.title('Logistic Regression Feature Importance', fontsize=16, fontweight='bold')
plt.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/plots/feature_importance.png', dpi=300, bbox_inches='tight')
plt.close()

# Confusion Matrix
print("\n9. CONFUSION MATRIX")
print("-"*50)

cm = confusion_matrix(y_test, y_pred)
print("\nConfusion Matrix:")
print(f"  True Negatives: {cm[0,0]}")
print(f"  False Positives: {cm[0,1]}")
print(f"  False Negatives: {cm[1,0]}")
print(f"  True Positives: {cm[1,1]}")

plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Variant', 'Reference'],
            yticklabels=['Variant', 'Reference'])
plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.tight_layout()
plt.savefig(f'{output_dir}/plots/confusion_matrix.png', dpi=300, bbox_inches='tight')
plt.close()

# ROC and PR Curves
print("\n10. ROC AND PR CURVES")
print("-"*50)

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 6))

# ROC Curve
fpr, tpr, _ = roc_curve(y_test, y_proba)
ax1.plot(fpr, tpr, linewidth=2, label=f'AUC = {auc_roc:.3f}')
ax1.plot([0, 1], [0, 1], 'k--', alpha=0.5)
ax1.set_xlabel('False Positive Rate', fontsize=12)
ax1.set_ylabel('True Positive Rate', fontsize=12)
ax1.set_title('ROC Curve', fontsize=14, fontweight='bold')
ax1.legend()
ax1.grid(True, alpha=0.3)

# PR Curve
precision_curve, recall_curve, _ = precision_recall_curve(y_test, y_proba)
ax2.plot(recall_curve, precision_curve, linewidth=2, label=f'PR-AUC = {pr_auc:.3f}')
ax2.set_xlabel('Recall', fontsize=12)
ax2.set_ylabel('Precision', fontsize=12)
ax2.set_title('Precision-Recall Curve', fontsize=14, fontweight='bold')
ax2.legend()
ax2.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig(f'{output_dir}/plots/roc_pr_curves.png', dpi=300, bbox_inches='tight')
plt.close()

# Probability distribution analysis
print("\n11. PROBABILITY DISTRIBUTION ANALYSIS")
print("-"*50)

plt.figure(figsize=(10, 6))
plt.hist(y_proba[y_test == 0], bins=50, alpha=0.7, label='Variants', density=True, color='red')
plt.hist(y_proba[y_test == 1], bins=50, alpha=0.7, label='References', density=True, color='blue')
plt.xlabel('Predicted Probability of Being Reference', fontsize=12)
plt.ylabel('Density', fontsize=12)
plt.title('Distribution of Predicted Probabilities', fontsize=16, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig(f'{output_dir}/plots/probability_distribution.png', dpi=300, bbox_inches='tight')
plt.close()

# Error analysis
print("\n12. ERROR ANALYSIS")
print("-"*50)

# Find misclassified samples
misclassified_idx = np.where(y_pred != y_test)[0]
misclassified_test_idx = [test_idx[i] for i in misclassified_idx]

print(f"Total misclassifications: {len(misclassified_idx)} ({len(misclassified_idx)/len(y_test)*100:.1f}%)")

# Analyze false positives and false negatives
false_positives_idx = np.where((y_test == 0) & (y_pred == 1))[0]
false_negatives_idx = np.where((y_test == 1) & (y_pred == 0))[0]

print(f"  False Positives: {len(false_positives_idx)} (variants predicted as references)")
print(f"  False Negatives: {len(false_negatives_idx)} (references predicted as variants)")

# Feature values for misclassified samples
if len(misclassified_idx) > 0:
    misclassified_features = X_test[misclassified_idx]
    correctly_classified_features = X_test[y_pred == y_test]
    
    print("\nAverage feature values:")
    print("Feature | Misclassified | Correct | Difference")
    print("-" * 50)
    for i, feature in enumerate(feature_names):
        mis_mean = np.mean(misclassified_features[:, i])
        cor_mean = np.mean(correctly_classified_features[:, i])
        diff = mis_mean - cor_mean
        print(f"{feature:20s} | {mis_mean:10.3f} | {cor_mean:8.3f} | {diff:+.3f}")

# Save model and results
print("\n13. SAVING RESULTS")
print("-"*50)

# Save model
import joblib
joblib.dump(final_lr, f'{output_dir}/logistic_regression_model.pkl')
joblib.dump(scaler, f'{output_dir}/scaler.pkl')

# Save results summary
with open(f'{output_dir}/results_summary.txt', 'w') as f:
    f.write("="*80 + "\n")
    f.write("BALANCED DATASET LOGISTIC REGRESSION - RESULTS SUMMARY\n")
    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    f.write("="*80 + "\n\n")
    
    f.write("DATASET INFORMATION:\n")
    f.write(f"- Total sentence families: {len(sentence_families)}\n")
    f.write(f"- Families with complete data: {families_with_complete_data}\n")
    f.write(f"- Total samples: {len(balanced_data)}\n")
    f.write(f"- Class distribution: {np.sum(y == 0)} variants, {np.sum(y == 1)} references\n\n")
    
    f.write("MODEL INFORMATION:\n")
    f.write(f"- Best parameters: {best_params}\n")
    f.write(f"- Best CV AUC: {best_score:.4f}\n\n")
    
    f.write("TEST SET PERFORMANCE:\n")
    f.write(f"- Accuracy: {accuracy:.4f}\n")
    f.write(f"- Precision: {precision:.4f}\n")
    f.write(f"- Recall: {recall:.4f}\n")
    f.write(f"- F1-Score: {f1:.4f}\n")
    f.write(f"- AUC-ROC: {auc_roc:.4f}\n")
    f.write(f"- PR-AUC: {pr_auc:.4f}\n\n")
    
    f.write("FEATURE IMPORTANCE:\n")
    f.write(feature_importance.to_string(index=False))
    f.write("\n\nCONFUSION MATRIX:\n")
    f.write(f"TN: {cm[0,0]}, FP: {cm[0,1]}\n")
    f.write(f"FN: {cm[1,0]}, TP: {cm[1,1]}\n")

# Save detailed results
results_df = pd.DataFrame({
    'sentence_id': [sentence_ids[i] for i in test_idx],
    'family_id': [family_ids[i] for i in test_idx],
    'true_label': y_test,
    'predicted_label': y_pred,
    'predicted_probability': y_proba,
    'correct': y_test == y_pred
})

results_df.to_csv(f'{output_dir}/detailed_predictions.csv', index=False)

# Save hyperparameter search results
hp_results_df = pd.DataFrame(results)
hp_results_df.to_csv(f'{output_dir}/hyperparameter_search_results.csv', index=False)

print(f"\n✓ Complete analysis saved to: {output_dir}/")
print(f"  - Model: logistic_regression_model.pkl")
print(f"  - Scaler: scaler.pkl")
print(f"  - Results summary: results_summary.txt")
print(f"  - Detailed predictions: detailed_predictions.csv")
print(f"  - Plots: plots/")
print("\n" + "="*80)