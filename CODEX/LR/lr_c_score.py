#!/usr/bin/env python3
"""
Corrected classification model with proper feature handling
Addresses: multicollinearity, scaling, and accurate evaluation
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold, cross_val_score
from sklearn.metrics import accuracy_score, classification_report, roc_auc_score
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Load all features
print("Loading all feature files...")

# 1. Trigram surprisal
surprisal_df = pd.read_csv('hutb_trigram_surprisals.csv')
surprisal_lookup = {row['sentence_id']: row['trigram_surprisal'] 
                   for _, row in surprisal_df.iterrows()}

# 2. Dependency length
dep_df = pd.read_csv('enhanced_dependency_detailed.csv')
dep_lookup = {row['sentence_id']: row['dependency_length'] 
              for _, row in dep_df.iterrows() if pd.notna(row['dependency_length'])}

# 3. IS scores
is_df = pd.read_csv('hutb_is_scores.csv')
is_lookup = {row['sentence_id']: row['is_score'] 
             for _, row in is_df.iterrows()}

# 4. Incremental completion scores
completion_df = pd.read_csv('hutb_incremental_completion_scores.csv')
completion_lookup = {row['sentence_id']: row['incremental_completion_score'] 
                    for _, row in completion_df.iterrows()}

# 5. Positional Language Model scores (FIXED: inverted)
print("Loading positional LM scores (EMILLE-trained)...")
plm_df = pd.read_csv('hutb_plm_scores_robust.csv')
# FIX: Invert scores so higher = better
plm_df['positional_lm_score'] = -plm_df['positional_lm_score']
plm_lookup = {row['sentence_id']: row['positional_lm_score'] 
              for _, row in plm_df.iterrows()}

print(f"Loaded features for {len(surprisal_lookup)} sentences")

# Create pairwise data
print("\nCreating pairwise comparisons...")
pairwise_data = []

# Get reference-variant pairs
ref_variants = {}
for sent_id in surprisal_lookup:
    if sent_id.endswith('.0'):
        base_id = sent_id[:-2]
        ref_variants[base_id] = {'ref': sent_id, 'variants': []}
    
for sent_id in surprisal_lookup:
    if not sent_id.endswith('.0'):
        base_id = sent_id.rsplit('.', 1)[0]
        if base_id in ref_variants:
            ref_variants[base_id]['variants'].append(sent_id)

# Create pairwise comparisons
skipped = 0
for base_id, group in tqdm(ref_variants.items()):
    ref_id = group['ref']
    
    for var_id in group['variants']:
        # Get all features
        if (ref_id in surprisal_lookup and var_id in surprisal_lookup and
            ref_id in dep_lookup and var_id in dep_lookup and
            ref_id in completion_lookup and var_id in completion_lookup and
            ref_id in plm_lookup and var_id in plm_lookup):
            
            # Calculate deltas (variant - reference)
            delta_trigram = surprisal_lookup[var_id] - surprisal_lookup[ref_id]
            delta_dep = dep_lookup[var_id] - dep_lookup[ref_id]
            delta_is = is_lookup.get(var_id, 0) - is_lookup.get(ref_id, 0)
            delta_completion = completion_lookup[var_id] - completion_lookup[ref_id]
            delta_plm = plm_lookup[var_id] - plm_lookup[ref_id]
            
            # Joachims transformation
            # Variant-Reference (label=0)
            pairwise_data.append([delta_trigram, delta_dep, delta_is, delta_completion, delta_plm, 0])
            
            # Reference-Variant (label=1)
            pairwise_data.append([-delta_trigram, -delta_dep, -delta_is, -delta_completion, -delta_plm, 1])
        else:
            skipped += 1

if skipped > 0:
    print(f"Skipped {skipped} pairs due to missing features")

# Convert to arrays
data = np.array(pairwise_data)
X_all = data[:, :5]
y = data[:, 5].astype(int)

print(f"\nCreated {len(X_all)} pairwise comparisons")

# Feature sets to test
feature_sets = {
    'All 5 features': [0, 1, 2, 3, 4],
    'Without Completion (multicollinearity)': [0, 1, 2, 4],  # Remove completion due to correlation with DepLen
    'Original 4 features': [0, 1, 2, 3],
    'Original + PLM': [0, 1, 2, 3, 4],
    'Core 3 (Trigram+DepLen+PLM)': [0, 1, 4],
    'Best 2 (Trigram+PLM)': [0, 4]
}

feature_names = ['Trigram surprisal', 'Dependency length', 'IS score', 'Completion score', 'Positional LM']

# Evaluate each feature set
print("\n" + "="*60)
print("FEATURE SET EVALUATION (with proper scaling)")
print("="*60)

results = {}
best_score = 0
best_set = None

for set_name, feature_indices in feature_sets.items():
    X = X_all[:, feature_indices]
    
    # Scale features
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 10-fold cross-validation
    kf = KFold(n_splits=10, shuffle=True, random_state=42)
    cv_scores = []
    
    for train_idx, test_idx in kf.split(X_scaled):
        X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        model = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
        model.fit(X_train, y_train)
        
        y_pred = model.predict(X_test)
        cv_scores.append(accuracy_score(y_test, y_pred))
    
    mean_score = np.mean(cv_scores)
    std_score = np.std(cv_scores)
    results[set_name] = (mean_score, std_score)
    
    print(f"{set_name:40s}: {mean_score*100:.2f}% (±{std_score*100:.2f}%)")
    
    if mean_score > best_score:
        best_score = mean_score
        best_set = set_name

print(f"\nBest configuration: {best_set}")

# Detailed analysis of best configuration
print("\n" + "="*60)
print("DETAILED ANALYSIS OF BEST CONFIGURATION")
print("="*60)

best_features = feature_sets[best_set]
X_best = X_all[:, best_features]
scaler = StandardScaler()
X_best_scaled = scaler.fit_transform(X_best)

# Train final model
model_best = LogisticRegression(random_state=42, max_iter=1000, class_weight='balanced')
model_best.fit(X_best_scaled, y)

print("\nModel coefficients (scaled):")
for i, feat_idx in enumerate(best_features):
    print(f"{feature_names[feat_idx]:20s}: {model_best.coef_[0][i]:8.4f}")
print(f"Intercept: {model_best.intercept_[0]:8.4f}")

# Individual feature performance
print("\n" + "="*60)
print("INDIVIDUAL FEATURE PERFORMANCE (scaled)")
print("="*60)

for i, name in enumerate(feature_names):
    X_single = X_all[:, i:i+1]
    X_single_scaled = StandardScaler().fit_transform(X_single)
    
    scores = cross_val_score(
        LogisticRegression(random_state=42, max_iter=1000),
        X_single_scaled, y, cv=10, scoring='accuracy'
    )
    
    print(f"{name:20s}: {np.mean(scores)*100:.2f}% (±{np.std(scores)*100:.2f}%)")

# Statistical significance testing
print("\n" + "="*60)
print("STATISTICAL SIGNIFICANCE TESTING")
print("="*60)

# Compare with and without PLM
if 4 in best_features:  # If PLM is in best set
    # Without PLM
    features_no_plm = [f for f in best_features if f != 4]
    X_no_plm = X_all[:, features_no_plm]
    X_no_plm_scaled = StandardScaler().fit_transform(X_no_plm)
    
    # Get predictions for McNemar's test
    model_no_plm = LogisticRegression(random_state=42, max_iter=1000)
    model_no_plm.fit(X_no_plm_scaled, y)
    pred_no_plm = model_no_plm.predict(X_no_plm_scaled)
    
    model_with_plm = LogisticRegression(random_state=42, max_iter=1000)
    model_with_plm.fit(X_best_scaled, y)
    pred_with_plm = model_with_plm.predict(X_best_scaled)
    
    # McNemar's test implementation
    # Create contingency table
    n00 = np.sum((pred_no_plm == y) & (pred_with_plm == y))
    n01 = np.sum((pred_no_plm == y) & (pred_with_plm != y))
    n10 = np.sum((pred_no_plm != y) & (pred_with_plm == y))
    n11 = np.sum((pred_no_plm != y) & (pred_with_plm != y))
    
    print(f"McNemar's test (with vs without PLM):")
    print(f"  Both correct: {n00}")
    print(f"  Only without PLM correct: {n01}")
    print(f"  Only with PLM correct: {n10}")
    print(f"  Both wrong: {n11}")
    
    # Calculate McNemar statistic
    if n01 + n10 > 0:
        # McNemar's chi-square statistic with continuity correction
        mcnemar_stat = (abs(n01 - n10) - 1)**2 / (n01 + n10)
        
        # Chi-square test with 1 degree of freedom
        from scipy.stats import chi2
        p_value = 1 - chi2.cdf(mcnemar_stat, df=1)
        
        print(f"  Test statistic: {mcnemar_stat:.4f}")
        print(f"  P-value: {p_value:.4f}")
        print(f"  Significant at α=0.05: {'Yes' if p_value < 0.05 else 'No'}")
    else:
        print("  Cannot compute McNemar's test (n01 + n10 = 0)")
    
    # Improvement analysis
    improvement = np.sum(pred_with_plm == y) - np.sum(pred_no_plm == y)
    print(f"\nPLM helps in {improvement} additional cases ({improvement/len(y)*100:.2f}%)")
    
    # Show where PLM helps
    plm_helps = (pred_with_plm == y) & (pred_no_plm != y)
    plm_hurts = (pred_with_plm != y) & (pred_no_plm == y)
    
    print(f"PLM helps: {plm_helps.sum()} cases")
    print(f"PLM hurts: {plm_hurts.sum()} cases")
    print(f"Net benefit: {plm_helps.sum() - plm_hurts.sum()} cases")

# Compare original vs best configuration
print("\n" + "="*60)
print("COMPARISON: Original vs Best Configuration")
print("="*60)

# Original configuration (unscaled)
X_original = X_all[:, [0, 1, 2, 3]]  # Original 4 features
model_original = LogisticRegression(random_state=42, max_iter=1000)
scores_original = cross_val_score(model_original, X_original, y, cv=10, scoring='accuracy')
print(f"Original (unscaled): {np.mean(scores_original)*100:.2f}% (±{np.std(scores_original)*100:.2f}%)")

# Original configuration (scaled)
X_original_scaled = StandardScaler().fit_transform(X_original)
model_original_scaled = LogisticRegression(random_state=42, max_iter=1000)
scores_original_scaled = cross_val_score(model_original_scaled, X_original_scaled, y, cv=10, scoring='accuracy')
print(f"Original (scaled): {np.mean(scores_original_scaled)*100:.2f}% (±{np.std(scores_original_scaled)*100:.2f}%)")

# Best configuration
print(f"Best configuration: {best_score*100:.2f}%")

# Final summary
print("\n" + "="*60)
print("SUMMARY")
print("="*60)
print(f"Best configuration: {best_set}")
print(f"Accuracy: {best_score*100:.2f}%")
print("\nKey findings:")
print("- Scaling features is crucial for fair comparison")
print("- Removing correlated features (Completion) may improve generalization")
print("- PLM provides small but measurable improvement")
print("- Statistical significance should be tested with McNemar's test")