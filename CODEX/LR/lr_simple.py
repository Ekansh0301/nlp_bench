#!/usr/bin/env python3
"""
Final model with Case Marker feature added
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Load all features (including new Case Marker scores)
print("Loading all feature files...")

# Previous features...
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
plm_df['positional_lm_score'] = -plm_df['positional_lm_score']  # Fix inversion
plm_lookup = {row['sentence_id']: row['positional_lm_score'] 
              for _, row in plm_df.iterrows()}

# NEW: Case Marker scores
print("Loading case marker scores...")
cm_df = pd.read_csv('hutb_case_marker_scores.csv')
cm_lookup = {row['sentence_id']: row['case_marker_score'] 
             for _, row in cm_df.iterrows()}

# Create pairwise comparisons
print("\nCreating pairwise comparisons...")
pairwise_data = []

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

# Create features with Case Marker
for base_id, group in tqdm(ref_variants.items()):
    ref_id = group['ref']
    
    for var_id in group['variants']:
        if (ref_id in surprisal_lookup and var_id in surprisal_lookup and
            ref_id in dep_lookup and var_id in dep_lookup and
            ref_id in plm_lookup and var_id in plm_lookup and
            ref_id in cm_lookup and var_id in cm_lookup):
            
            # Calculate deltas
            delta_trigram = surprisal_lookup[var_id] - surprisal_lookup[ref_id]
            delta_dep = dep_lookup[var_id] - dep_lookup[ref_id]
            delta_is = is_lookup.get(var_id, 0) - is_lookup.get(ref_id, 0)
            delta_plm = plm_lookup[var_id] - plm_lookup[ref_id]
            delta_cm = cm_lookup[var_id] - cm_lookup[ref_id]  # NEW
            
            # Joachims transformation
            pairwise_data.append([delta_trigram, delta_dep, delta_is, delta_plm, delta_cm, 0])
            pairwise_data.append([-delta_trigram, -delta_dep, -delta_is, -delta_plm, -delta_cm, 1])

data = np.array(pairwise_data)
X = data[:, :5]
y = data[:, 5].astype(int)

print(f"\nCreated {len(X)} pairwise comparisons")

# Scale and train
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# 10-fold CV
kf = KFold(n_splits=10, shuffle=True, random_state=42)
cv_scores = []

for train_idx, test_idx in kf.split(X_scaled):
    X_train, X_test = X_scaled[train_idx], X_scaled[test_idx]
    y_train, y_test = y[train_idx], y[test_idx]
    
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)
    
    y_pred = model.predict(X_test)
    cv_scores.append(accuracy_score(y_test, y_pred))

print("\n" + "="*60)
print("RESULTS WITH CASE MARKER FEATURE")
print("="*60)
print(f"Mean accuracy: {np.mean(cv_scores)*100:.2f}%")
print(f"Std deviation: {np.std(cv_scores)*100:.2f}%")

# Compare with previous best (without Case Marker)
X_prev = X[:, [0, 1, 2, 3]]  # Without Case Marker
X_prev_scaled = scaler.fit_transform(X_prev)

cv_scores_prev = []
for train_idx, test_idx in kf.split(X_prev_scaled):
    model_prev = LogisticRegression(random_state=42, max_iter=1000)
    model_prev.fit(X_prev_scaled[train_idx], y[train_idx])
    cv_scores_prev.append(accuracy_score(y[test_idx], model_prev.predict(X_prev_scaled[test_idx])))

print(f"\nPrevious best (without CM): {np.mean(cv_scores_prev)*100:.2f}%")
print(f"With Case Marker: {np.mean(cv_scores)*100:.2f}%")
print(f"Improvement: {(np.mean(cv_scores) - np.mean(cv_scores_prev))*100:.2f}%")

# Feature importance
model_full = LogisticRegression(random_state=42, max_iter=1000)
model_full.fit(X_scaled, y)

feature_names = ['Trigram', 'DepLen', 'IS', 'PLM', 'CaseMarker']
print("\nFeature coefficients (scaled):")
for i, name in enumerate(feature_names):
    print(f"{name:12s}: {model_full.coef_[0][i]:.4f}")