#!/usr/bin/env python3
"""
Logistic Regression with UID_Slope and Core Features (DepLen + UID_Std removed)
Includes Full Ablation Analysis
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm

# Load features
print("Loading feature files...")

surprisal_df = pd.read_csv('DATA/hutb_trigram_surprisals.csv')
dep_df = pd.read_csv('DATA/enhanced_dependency_detailed.csv')  # still used for safety
is_df = pd.read_csv('DATA/hutb_is_scores.csv')
plm_df = pd.read_csv('DATA/hutb_plm_scores_robust.csv')
cm_df = pd.read_csv('DATA/hutb_case_marker_scores.csv')
uid_df = pd.read_csv('DATA/hutb_uid_scores.csv')  # Contains: Sentence ID, uid_slope

plm_df['positional_lm_score'] = -plm_df['positional_lm_score']

# Create lookups
surprisal_lookup = {r['sentence_id']: r['trigram_surprisal'] for _, r in surprisal_df.iterrows()}
is_lookup        = {r['sentence_id']: r['is_score'] for _, r in is_df.iterrows()}
plm_lookup       = {r['sentence_id']: r['positional_lm_score'] for _, r in plm_df.iterrows()}
cm_lookup        = {r['sentence_id']: r['case_marker_score'] for _, r in cm_df.iterrows()}
uid_lookup       = {r['Sentence ID']: r['uid_slope'] for _, r in uid_df.iterrows()}

# Build reference-variant pairs
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

# Construct deltas
for base_id, group in tqdm(ref_variants.items()):
    ref_id = group['ref']
    for var_id in group['variants']:
        if all(k in d for k, d in [
            (ref_id, surprisal_lookup), (var_id, surprisal_lookup),
            (ref_id, is_lookup),        (var_id, is_lookup),
            (ref_id, plm_lookup),       (var_id, plm_lookup),
            (ref_id, cm_lookup),        (var_id, cm_lookup),
            (ref_id, uid_lookup),       (var_id, uid_lookup)
        ]):
            delta_trigram    = surprisal_lookup[var_id] - surprisal_lookup[ref_id]
            delta_is         = is_lookup.get(var_id, 0) - is_lookup.get(ref_id, 0)
            delta_plm        = plm_lookup[var_id] - plm_lookup[ref_id]
            delta_cm         = cm_lookup[var_id] - cm_lookup[ref_id]
            delta_uid_slope  = uid_lookup[var_id] - uid_lookup[ref_id]

            features = [
                delta_trigram, delta_is, delta_plm, delta_cm, delta_uid_slope
            ]

            pairwise_data.append(features + [0])
            pairwise_data.append([-x for x in features] + [1])

# Prepare data
data = np.array(pairwise_data)
X = data[:, :-1]
y = data[:, -1].astype(int)

# Feature names
feature_names = [
    "Trigram", "IS", "PLM", "CaseMarker", "UID_Slope"
]

# Scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# KFold
kf = StratifiedKFold(n_splits=10, shuffle=True, random_state=42)

def evaluate_model(X_subset, y):
    accs = []
    for train_idx, test_idx in kf.split(X_subset,y):
        model = LogisticRegression(random_state=42, max_iter=1000)
        model.fit(X_subset[train_idx], y[train_idx])
        preds = model.predict(X_subset[test_idx])
        accs.append(accuracy_score(y[test_idx], preds))
    return np.mean(accs), np.std(accs)

# === Evaluation ===
print("\n" + "="*60)
print("LOGISTIC REGRESSION WITH UID_Slope AND CORE FEATURES")
print("="*60)

acc, std = evaluate_model(X_scaled, y)
print(f"{'Final Accuracy':25s} Accuracy: {acc*100:.2f}% ± {std*100:.2f}%")

# === Leave-one-out ablation ===
print("\nLEAVE-ONE-OUT ABLATION (Remove 1 feature)")
for i, name in enumerate(feature_names):
    X_ablated = np.delete(X_scaled, i, axis=1)
    acc, std = evaluate_model(X_ablated, y)
    print(f"Without {name:17s} Accuracy: {acc*100:.2f}% ± {std*100:.2f}%")

# === Single feature ===
print("\nSINGLE FEATURE PERFORMANCE (Only 1 feature)")
for i, name in enumerate(feature_names):
    X_single = X_scaled[:, i].reshape(-1, 1)
    acc, std = evaluate_model(X_single, y)
    print(f"{name:22s} Only Accuracy: {acc*100:.2f}% ± {std*100:.2f}%")

# === Feature importance ===
print("\n" + "="*60)
print("FEATURE IMPORTANCE")
print("="*60)

model = LogisticRegression(random_state=42, max_iter=1000)
model.fit(X_scaled, y)

for i, name in enumerate(feature_names):
    print(f"{name:12s}: {model.coef_[0][i]: .4f}")
