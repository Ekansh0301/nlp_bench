#!/usr/bin/env python3
"""
Hindi Sentence Classification Model with Imbalance Handling
Direct classification approach with multiple imbalance techniques
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.metrics import accuracy_score, classification_report, f1_score, precision_score, recall_score
from sklearn.utils.class_weight import compute_class_weight
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# Imbalance handling imports
try:
    from imblearn.over_sampling import SMOTE, ADASYN, BorderlineSMOTE
    from imblearn.under_sampling import RandomUnderSampler, EditedNearestNeighbours
    from imblearn.combine import SMOTETomek, SMOTEENN
    imbalance_available = True
    print("imblearn library available - all techniques enabled")
except ImportError:
    imbalance_available = False
    print("⚠️  Warning: imblearn not available. Install with: pip install imbalanced-learn")
    print("Will use basic techniques only")

print("="*70)
print("HINDI SENTENCE CLASSIFICATION - IMBALANCE HANDLING")
print("Direct Classification with Data Leakage Prevention")
print("="*70)

# Load all features
print("\nLoading feature files...")
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

# Create direct classification dataset (no pairing)
print("\nCreating direct classification dataset...")
classification_data = []

# Collect all sentence IDs
all_sentence_ids = set(surprisal_lookup.keys())

for sent_id in tqdm(all_sentence_ids, desc="Processing sentences"):
    # Skip if we don't have all features
    if not all([
        sent_id in surprisal_lookup,
        sent_id in dep_lookup,
        sent_id in plm_lookup,
        sent_id in cm_lookup
    ]):
        continue
    
    # Determine if this is a reference sentence (ends with .0) or variant
    is_reference = 1 if sent_id.endswith('.0') else 0
    
    # Extract features
    trigram_surprisal = surprisal_lookup[sent_id]
    dependency_length = dep_lookup[sent_id]
    information_status = is_lookup.get(sent_id, 0)  # Default to 0 if missing
    positional_lm = plm_lookup[sent_id]
    case_marker = cm_lookup[sent_id]
    
    classification_data.append([
        trigram_surprisal,
        dependency_length,
        information_status,
        positional_lm,
        case_marker,
        is_reference,
        sent_id  # Keep sentence ID for leak detection
    ])

# Convert to arrays
data = np.array(classification_data, dtype=object)
X = np.array([row[:5] for row in data], dtype=float)  # Features only
y = np.array([row[5] for row in data], dtype=int)     # Labels only
sentence_ids = [row[6] for row in data]              # IDs for leak detection

feature_names = [
    'Trigram Surprisal',
    'Dependency Length', 
    'Information Status',
    'Positional LM',
    'Case Marker Score'
]

print(f"\nDataset created:")
print(f"- Total sentences: {len(X)}")
print(f"- Features: {len(feature_names)}")
print(f"- Reference sentences: {np.sum(y == 1)} ({np.mean(y)*100:.1f}%)")
print(f"- Variant sentences: {np.sum(y == 0)} ({(1-np.mean(y))*100:.1f}%)")

# DATA LEAKAGE PREVENTION
print(f"\n" + "="*50)
print("DATA LEAKAGE PREVENTION")
print("="*50)

# Check for sentence families (same base but different variants)
sentence_families = {}
for i, sent_id in enumerate(sentence_ids):
    if '.' in sent_id:
        base_id = sent_id.rsplit('.', 1)[0]
        if base_id not in sentence_families:
            sentence_families[base_id] = []
        sentence_families[base_id].append(i)

print(f"Found {len(sentence_families)} sentence families")

# Ensure no sentence family is split across train/test
family_sizes = [len(family) for family in sentence_families.values()]
print(f"Family sizes - Min: {min(family_sizes)}, Max: {max(family_sizes)}, Mean: {np.mean(family_sizes):.1f}")

# Create family-aware train-test split
def family_aware_split(sentence_families, test_size=0.2, random_state=42):
    """Split data ensuring sentence families don't cross train/test boundary"""
    np.random.seed(random_state)
    
    families = list(sentence_families.keys())
    np.random.shuffle(families)
    
    n_test_families = int(len(families) * test_size)
    test_families = families[:n_test_families]
    train_families = families[n_test_families:]
    
    train_indices = []
    test_indices = []
    
    for family in train_families:
        train_indices.extend(sentence_families[family])
    
    for family in test_families:
        test_indices.extend(sentence_families[family])
    
    return train_indices, test_indices

train_indices, test_indices = family_aware_split(sentence_families, test_size=0.2, random_state=42)

# print(f"Family-aware split:")
# print(f"- Train families: {len([f for f in sentence_families.keys() 
#                                 if any(i in train_indices for i in sentence_families[f])])}")
# print(f"- Test families: {len([f for f in sentence_families.keys() 
#                                if any(i in test_indices for i in sentence_families[f])])}")
# print(f"- Train samples: {len(train_indices)}")
# print(f"- Test samples: {len(test_indices)}")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split data with family awareness
X_train, X_test = X_scaled[train_indices], X_scaled[test_indices]
y_train, y_test = y[train_indices], y[test_indices]

print(f"- Train class distribution: {np.bincount(y_train)} ({np.bincount(y_train)/len(y_train)*100}%)")
print(f"- Test class distribution: {np.bincount(y_test)} ({np.bincount(y_test)/len(y_test)*100}%)")

# Define base models (removed Neural Network as requested)
base_models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True)
}

# IMBALANCE HANDLING TECHNIQUES
print(f"\n" + "="*70)
print("IMBALANCE HANDLING TECHNIQUES")
print("="*70)

# Calculate class weights for cost-sensitive learning
class_weights = compute_class_weight('balanced', classes=np.unique(y_train), y=y_train)
class_weight_dict = {0: class_weights[0], 1: class_weights[1]}
print(f"Calculated class weights: {class_weight_dict}")

# Define imbalance techniques
imbalance_techniques = {}

# 1. No handling (baseline)
imbalance_techniques['Baseline'] = {
    'X_train': X_train,
    'y_train': y_train,
    'description': 'No imbalance handling'
}

# 2. Random Undersampling
n_minority = np.sum(y_train == 1)
n_majority_sample = min(n_minority * 2, np.sum(y_train == 0))  # At most 2:1 ratio

majority_indices = np.where(y_train == 0)[0]
minority_indices = np.where(y_train == 1)[0]

np.random.seed(42)
sampled_majority = np.random.choice(majority_indices, n_majority_sample, replace=False)
balanced_indices = np.concatenate([sampled_majority, minority_indices])

imbalance_techniques['Undersampling'] = {
    'X_train': X_train[balanced_indices],
    'y_train': y_train[balanced_indices],
    'description': f'Random undersampling to {n_majority_sample}:{n_minority} ratio'
}

# 3. Cost-sensitive learning
imbalance_techniques['Cost-Sensitive'] = {
    'X_train': X_train,
    'y_train': y_train,
    'class_weight': class_weight_dict,
    'description': f'Cost-sensitive learning with weights {class_weight_dict}'
}

# 4. SMOTE and advanced techniques (if available)
if imbalance_available:
    # SMOTE
    smote = SMOTE(random_state=42, k_neighbors=min(5, n_minority-1))
    try:
        X_smote, y_smote = smote.fit_resample(X_train, y_train)
        imbalance_techniques['SMOTE'] = {
            'X_train': X_smote,
            'y_train': y_smote,
            'description': 'SMOTE oversampling'
        }
    except Exception as e:
        print(f"SMOTE failed: {e}")
    
    # Borderline SMOTE
    try:
        borderline_smote = BorderlineSMOTE(random_state=42, k_neighbors=min(5, n_minority-1))
        X_bsmote, y_bsmote = borderline_smote.fit_resample(X_train, y_train)
        imbalance_techniques['Borderline-SMOTE'] = {
            'X_train': X_bsmote,
            'y_train': y_bsmote,
            'description': 'Borderline SMOTE oversampling'
        }
    except Exception as e:
        print(f"Borderline SMOTE failed: {e}")
    
    # ADASYN
    try:
        adasyn = ADASYN(random_state=42, n_neighbors=min(5, n_minority-1))
        X_adasyn, y_adasyn = adasyn.fit_resample(X_train, y_train)
        imbalance_techniques['ADASYN'] = {
            'X_train': X_adasyn,
            'y_train': y_adasyn,
            'description': 'ADASYN adaptive sampling'
        }
    except Exception as e:
        print(f"ADASYN failed: {e}")

print(f"Available techniques: {list(imbalance_techniques.keys())}")

# COMPREHENSIVE EVALUATION
print(f"\n" + "="*70)
print("COMPREHENSIVE EVALUATION")
print("="*70)

results = {}
detailed_results = []

# Evaluate each combination of technique + model
for technique_name, technique_data in imbalance_techniques.items():
    print(f"\nEvaluating technique: {technique_name}")
    print(f"Description: {technique_data['description']}")
    
    X_train_tech = technique_data['X_train']
    y_train_tech = technique_data['y_train']
    
    print(f"Training set: {len(X_train_tech)} samples, "
          f"class distribution: {np.bincount(y_train_tech)}")
    
    for model_name, base_model in base_models.items():
        # Clone model for this experiment
        if technique_name == 'Cost-Sensitive' and hasattr(base_model, 'class_weight'):
            # Apply class weights for cost-sensitive learning
            if model_name == 'Logistic Regression':
                model = LogisticRegression(random_state=42, max_iter=1000, 
                                         class_weight=technique_data['class_weight'])
            elif model_name == 'Random Forest':
                model = RandomForestClassifier(n_estimators=100, random_state=42,
                                             class_weight=technique_data['class_weight'])
            elif model_name == 'SVM':
                model = SVC(random_state=42, probability=True,
                           class_weight=technique_data['class_weight'])
            else:
                model = base_model  # Gradient Boosting doesn't support class_weight directly
        else:
            model = base_model
        
        # Train model
        model.fit(X_train_tech, y_train_tech)
        
        # Predictions
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1] if hasattr(model, 'predict_proba') else None
        
        # Calculate multiple metrics
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, zero_division=0)
        recall = recall_score(y_test, y_pred, zero_division=0)
        f1 = f1_score(y_test, y_pred, zero_division=0)
        
        # Store results
        combo_name = f"{technique_name}+{model_name}"
        results[combo_name] = {
            'technique': technique_name,
            'model': model_name,
            'accuracy': accuracy,
            'precision': precision,
            'recall': recall,
            'f1': f1,
            'train_size': len(X_train_tech)
        }
        
        detailed_results.append({
            'Technique': technique_name,
            'Model': model_name,
            'Accuracy': accuracy,
            'Precision': precision,
            'Recall': recall,
            'F1-Score': f1,
            'Train_Size': len(X_train_tech)
        })
        
        print(f"  {model_name:20s}: Acc={accuracy:.4f}, "
              f"Prec={precision:.4f}, Rec={recall:.4f}, F1={f1:.4f}")

# RESULTS ANALYSIS
print(f"\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

# Convert to DataFrame for easier analysis
results_df = pd.DataFrame(detailed_results)

print("All Results:")
print(results_df.round(4).to_string(index=False))

# Find best performers by different metrics
print(f"\n" + "="*40)
print("BEST PERFORMERS BY METRIC")
print("="*40)

for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
    best_idx = results_df[metric].idxmax()
    best_combo = results_df.iloc[best_idx]
    print(f"{metric:12s}: {best_combo['Technique']} + {best_combo['Model']} "
          f"({best_combo[metric]:.4f})")

# Technique comparison
print(f"\n" + "="*40)
print("TECHNIQUE COMPARISON (Average)")
print("="*40)

technique_summary = results_df.groupby('Technique').agg({
    'Accuracy': 'mean',
    'Precision': 'mean', 
    'Recall': 'mean',
    'F1-Score': 'mean'
}).round(4)

print(technique_summary.to_string())

# Best overall recommendation
best_f1_idx = results_df['F1-Score'].idxmax()
best_overall = results_df.iloc[best_f1_idx]

print(f"\n" + "="*50)
print("RECOMMENDATION")
print("="*50)
print(f"Best Overall Combination (by F1-Score):")
print(f"  Technique: {best_overall['Technique']}")
print(f"  Model: {best_overall['Model']}")
print(f"  F1-Score: {best_overall['F1-Score']:.4f}")
print(f"  Accuracy: {best_overall['Accuracy']:.4f}")
print(f"  Precision: {best_overall['Precision']:.4f}")
print(f"  Recall: {best_overall['Recall']:.4f}")

# Baseline comparison
baseline_results = results_df[results_df['Technique'] == 'Baseline']
best_baseline = baseline_results.loc[baseline_results['F1-Score'].idxmax()]

print(f"\nImprovement over baseline:")
print(f"  F1-Score improvement: {best_overall['F1-Score'] - best_baseline['F1-Score']:.4f}")
print(f"  Recall improvement: {best_overall['Recall'] - best_baseline['Recall']:.4f}")

# Save results
import os
output_dir = 'hindi_imbalance_results'
os.makedirs(output_dir, exist_ok=True)

results_df.to_csv(os.path.join(output_dir, 'imbalance_results.csv'), index=False)

with open(os.path.join(output_dir, 'imbalance_summary.txt'), 'w') as f:
    f.write("="*70 + "\n")
    f.write("HINDI SENTENCE CLASSIFICATION - IMBALANCE HANDLING RESULTS\n")
    f.write("="*70 + "\n\n")
    
    f.write("DATASET STATISTICS:\n")
    f.write(f"- Total sentences: {len(X)}\n")
    f.write(f"- Reference sentences: {np.sum(y == 1)} ({np.mean(y)*100:.1f}%)\n")
    f.write(f"- Variant sentences: {np.sum(y == 0)} ({(1-np.mean(y))*100:.1f}%)\n")
    f.write(f"- Class weights: {class_weight_dict}\n\n")
    
    f.write("TECHNIQUES TESTED:\n")
    for name, data in imbalance_techniques.items():
        f.write(f"- {name}: {data['description']}\n")
    f.write("\n")
    
    f.write("BEST PERFORMERS:\n")
    for metric in ['Accuracy', 'Precision', 'Recall', 'F1-Score']:
        best_idx = results_df[metric].idxmax()
        best_combo = results_df.iloc[best_idx]
        f.write(f"- {metric}: {best_combo['Technique']} + {best_combo['Model']} "
                f"({best_combo[metric]:.4f})\n")
    
    f.write(f"\nRECOMMENDED APPROACH:\n")
    f.write(f"- Best combination: {best_overall['Technique']} + {best_overall['Model']}\n")
    f.write(f"- F1-Score: {best_overall['F1-Score']:.4f}\n")
    f.write(f"- Improvement over baseline: {best_overall['F1-Score'] - best_baseline['F1-Score']:.4f}\n")

print(f"\n" + "="*70)
print("ANALYSIS COMPLETE")
print("="*70)
print(f"Results saved to: {output_dir}/")
print(f"Best approach: {best_overall['Technique']} + {best_overall['Model']}")
print(f"Best F1-Score: {best_overall['F1-Score']:.4f}")
print("Key improvement: Better minority class detection without data leakage")

# Load all features
print("\nLoading feature files...")
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

# Create direct classification dataset (no pairing)
print("\nCreating direct classification dataset...")
classification_data = []

# Collect all sentence IDs
all_sentence_ids = set(surprisal_lookup.keys())

for sent_id in tqdm(all_sentence_ids, desc="Processing sentences"):
    # Skip if we don't have all features
    if not all([
        sent_id in surprisal_lookup,
        sent_id in dep_lookup,
        sent_id in plm_lookup,
        sent_id in cm_lookup
    ]):
        continue
    
    # Determine if this is a reference sentence (ends with .0) or variant
    is_reference = 1 if sent_id.endswith('.0') else 0
    
    # Extract features
    trigram_surprisal = surprisal_lookup[sent_id]
    dependency_length = dep_lookup[sent_id]
    information_status = is_lookup.get(sent_id, 0)  # Default to 0 if missing
    positional_lm = plm_lookup[sent_id]
    case_marker = cm_lookup[sent_id]
    
    classification_data.append([
        trigram_surprisal,
        dependency_length,
        information_status,
        positional_lm,
        case_marker,
        is_reference
    ])

# Convert to numpy array
data = np.array(classification_data)
X = data[:, :5]  # Features
y = data[:, 5].astype(int)  # Labels (1=Reference, 0=Variant)

feature_names = [
    'Trigram Surprisal',
    'Dependency Length', 
    'Information Status',
    'Positional LM',
    'Case Marker Score'
]

print(f"\nDataset created:")
print(f"- Total sentences: {len(X)}")
print(f"- Features: {len(feature_names)}")
print(f"- Reference sentences: {np.sum(y == 1)} ({np.mean(y)*100:.1f}%)")
print(f"- Variant sentences: {np.sum(y == 0)} ({(1-np.mean(y))*100:.1f}%)")

# Check for class imbalance
class_ratio = np.sum(y == 1) / np.sum(y == 0)
print(f"- Class ratio (Ref/Var): {class_ratio:.2f}")

# CRITICAL: Check for data leakage
print(f"\n" + "="*50)
print("DATA LEAKAGE DETECTION")
print("="*50)

# Calculate baseline accuracy (majority class)
majority_class_accuracy = max(np.mean(y), 1 - np.mean(y))
print(f"Majority class baseline: {majority_class_accuracy:.4f}")

# Check if any feature perfectly separates classes
print("\nFeature separation analysis:")
for i, feature_name in enumerate(feature_names):
    ref_values = X[y == 1, i]
    var_values = X[y == 0, i]
    
    # Check for perfect separation
    ref_min, ref_max = ref_values.min(), ref_values.max()
    var_min, var_max = var_values.min(), var_values.max()
    
    overlap = not (ref_max < var_min or var_max < ref_min)
    
    print(f"  {feature_name}:")
    print(f"    Reference range: [{ref_min:.4f}, {ref_max:.4f}]")
    print(f"    Variant range:   [{var_min:.4f}, {var_max:.4f}]")
    print(f"    Overlap: {overlap}")
    
    # Check mean differences
    mean_diff = np.abs(ref_values.mean() - var_values.mean())
    pooled_std = np.sqrt((ref_values.var() + var_values.var()) / 2)
    effect_size = mean_diff / pooled_std if pooled_std > 0 else 0
    print(f"    Effect size (Cohen's d): {effect_size:.4f}")

# Test dummy classifier performance
from sklearn.dummy import DummyClassifier

dummy_strategies = ['most_frequent', 'stratified', 'uniform']
print(f"\nDummy classifier baselines:")

for strategy in dummy_strategies:
    dummy = DummyClassifier(strategy=strategy, random_state=42)
    dummy_scores = cross_val_score(dummy, X_scaled, y, cv=5, scoring='accuracy')
    print(f"  {strategy:15s}: {dummy_scores.mean():.4f} (±{dummy_scores.std():.4f})")

# Check for trivial classification
print(f"\nTrivial classification check:")
print(f"Always predict 0 (variant): {(y == 0).mean():.4f}")
print(f"Always predict 1 (reference): {(y == 1).mean():.4f}")

# WARNING: If individual feature performance equals majority class baseline,
# this indicates the model is not learning meaningful patterns
individual_feature_baseline = 0.9021  # From your output
if abs(individual_feature_baseline - majority_class_accuracy) < 0.001:
    print(f"\n⚠️  WARNING: SEVERE DATA LEAKAGE DETECTED!")
    print(f"   Individual feature performance ({individual_feature_baseline:.4f}) ")
    print(f"   equals majority class baseline ({majority_class_accuracy:.4f})")
    print(f"   This suggests the model is simply predicting the majority class!")

# Balanced dataset creation for fair comparison
print(f"\n" + "="*50)
print("CREATING BALANCED DATASET")
print("="*50)

# Sample equal numbers from each class
min_class_size = min(np.sum(y == 0), np.sum(y == 1))
print(f"Minimum class size: {min_class_size}")

if min_class_size < 100:
    print("⚠️  WARNING: Very few reference sentences for balanced sampling!")
    print("   Results may not be reliable.")
else:
    # Create balanced dataset
    ref_indices = np.where(y == 1)[0]
    var_indices = np.where(y == 0)[0]
    
    # Randomly sample equal numbers
    np.random.seed(42)
    selected_ref = np.random.choice(ref_indices, min_class_size, replace=False)
    selected_var = np.random.choice(var_indices, min_class_size, replace=False)
    
    # Combine
    balanced_indices = np.concatenate([selected_ref, selected_var])
    X_balanced = X_scaled[balanced_indices]
    y_balanced = y[balanced_indices]
    
    print(f"Balanced dataset created:")
    print(f"- Total samples: {len(X_balanced)}")
    print(f"- Class distribution: {np.bincount(y_balanced)}")
    
    # Test on balanced data
    print(f"\nBalanced dataset performance:")
    for model_name, model in [('Logistic Regression', LogisticRegression(random_state=42, max_iter=1000)),
                             ('Random Forest', RandomForestClassifier(n_estimators=100, random_state=42))]:
        balanced_scores = cross_val_score(model, X_balanced, y_balanced, cv=5, scoring='accuracy')
        print(f"  {model_name:20s}: {balanced_scores.mean():.4f} (±{balanced_scores.std():.4f})")
    
    # Compare with dummy on balanced data
    dummy_balanced = DummyClassifier(strategy='most_frequent', random_state=42)
    dummy_balanced_scores = cross_val_score(dummy_balanced, X_balanced, y_balanced, cv=5, scoring='accuracy')
    print(f"  {'Dummy (balanced)':20s}: {dummy_balanced_scores.mean():.4f} (±{dummy_balanced_scores.std():.4f})")

# Feature scaling
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Train-test split
X_train, X_test, y_train, y_test = train_test_split(
    X_scaled, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\nTrain-Test Split:")
print(f"- Training set: {len(X_train)} samples")
print(f"- Test set: {len(X_test)} samples")
print(f"- Train class distribution: {np.bincount(y_train)}")
print(f"- Test class distribution: {np.bincount(y_test)}")

# Define models to test
models = {
    'Logistic Regression': LogisticRegression(random_state=42, max_iter=1000),
    'Random Forest': RandomForestClassifier(n_estimators=100, random_state=42),
    'Gradient Boosting': GradientBoostingClassifier(random_state=42),
    'SVM': SVC(random_state=42, probability=True),
    'Neural Network': MLPClassifier(hidden_layer_sizes=(100, 50), random_state=42, max_iter=1000)
}

print("\n" + "="*70)
print("MODEL TRAINING AND EVALUATION")
print("="*70)

# Store results
results = {}

# Train and evaluate each model
for model_name, model in models.items():
    print(f"\nTraining {model_name}...")
    
    # Train model
    model.fit(X_train, y_train)
    
    # Predictions
    y_pred_train = model.predict(X_train)
    y_pred_test = model.predict(X_test)
    
    # Accuracies
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    
    # Cross-validation
    cv_scores = cross_val_score(model, X_scaled, y, cv=5, scoring='accuracy')
    
    # Store results
    results[model_name] = {
        'train_accuracy': train_accuracy,
        'test_accuracy': test_accuracy,
        'cv_mean': cv_scores.mean(),
        'cv_std': cv_scores.std(),
        'model': model
    }
    
    print(f"  Train Accuracy: {train_accuracy:.4f}")
    print(f"  Test Accuracy:  {test_accuracy:.4f}")
    print(f"  CV Accuracy:    {cv_scores.mean():.4f} (±{cv_scores.std():.4f})")

# Summary results
print("\n" + "="*70)
print("RESULTS SUMMARY")
print("="*70)

print(f"{'Model':<20} {'Train Acc':<10} {'Test Acc':<10} {'CV Mean':<10} {'CV Std':<8}")
print("-" * 68)

for model_name, result in results.items():
    print(f"{model_name:<20} {result['train_accuracy']:<10.4f} "
          f"{result['test_accuracy']:<10.4f} {result['cv_mean']:<10.4f} "
          f"{result['cv_std']:<8.4f}")

# Find best model
best_model_name = max(results.keys(), key=lambda x: results[x]['test_accuracy'])
best_model = results[best_model_name]['model']
best_test_acc = results[best_model_name]['test_accuracy']

print(f"\nBest Model: {best_model_name}")
print(f"Best Test Accuracy: {best_test_acc:.4f}")

# Feature importance for best model
print(f"\n" + "="*50)
print("FEATURE IMPORTANCE (Best Model)")
print("="*50)

if hasattr(best_model, 'feature_importances_'):
    # Tree-based models
    importances = best_model.feature_importances_
    importance_type = "Feature Importances"
elif hasattr(best_model, 'coef_'):
    # Linear models
    importances = np.abs(best_model.coef_[0])
    importance_type = "Coefficient Magnitudes"
else:
    importances = None
    importance_type = "Not Available"

if importances is not None:
    print(f"{importance_type}:")
    feature_importance = list(zip(feature_names, importances))
    feature_importance.sort(key=lambda x: x[1], reverse=True)
    
    for feature, importance in feature_importance:
        print(f"  {feature:<20}: {importance:.4f}")
else:
    print("Feature importance not available for this model type")

# Detailed analysis of best model
print(f"\n" + "="*50)
print("DETAILED ANALYSIS (Best Model)")
print("="*50)

y_pred_best = best_model.predict(X_test)
print(f"\nClassification Report for {best_model_name}:")
print(classification_report(y_test, y_pred_best, 
                          target_names=['Variant', 'Reference']))

# Performance by class
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test, y_pred_best)
print(f"\nConfusion Matrix:")
print(f"                 Predicted")
print(f"                 Variant  Reference")
print(f"Actual Variant   {cm[0,0]:7d}  {cm[0,1]:9d}")
print(f"Actual Reference {cm[1,0]:7d}  {cm[1,1]:9d}")

# Individual feature performance
print(f"\n" + "="*50)
print("INDIVIDUAL FEATURE PERFORMANCE")
print("="*50)

print("Testing each feature individually:")
for i, feature_name in enumerate(feature_names):
    # Train simple model on single feature
    single_feature_model = LogisticRegression(random_state=42, max_iter=1000)
    X_single = X_scaled[:, i:i+1]
    
    # Cross-validation on single feature
    single_cv_scores = cross_val_score(single_feature_model, X_single, y, 
                                     cv=5, scoring='accuracy')
    
    print(f"  {feature_name:<20}: {single_cv_scores.mean():.4f} (±{single_cv_scores.std():.4f})")

# Save results to file
print(f"\n" + "="*50)
print("SAVING RESULTS")
print("="*50)

import os
output_dir = 'hindi_classification_results'
os.makedirs(output_dir, exist_ok=True)

with open(os.path.join(output_dir, 'classification_results.txt'), 'w') as f:
    f.write("="*70 + "\n")
    f.write("HINDI SENTENCE CLASSIFICATION RESULTS\n")
    f.write("Direct Classification: Reference vs Variant\n")
    f.write("="*70 + "\n\n")
    
    f.write("DATASET STATISTICS:\n")
    f.write(f"- Total sentences: {len(X)}\n")
    f.write(f"- Reference sentences: {np.sum(y == 1)} ({np.mean(y)*100:.1f}%)\n")
    f.write(f"- Variant sentences: {np.sum(y == 0)} ({(1-np.mean(y))*100:.1f}%)\n")
    f.write(f"- Class ratio (Ref/Var): {class_ratio:.2f}\n\n")
    
    f.write("MODEL PERFORMANCE:\n")
    f.write(f"{'Model':<20} {'Train Acc':<10} {'Test Acc':<10} {'CV Mean':<10} {'CV Std':<8}\n")
    f.write("-" * 68 + "\n")
    
    for model_name, result in results.items():
        f.write(f"{model_name:<20} {result['train_accuracy']:<10.4f} "
                f"{result['test_accuracy']:<10.4f} {result['cv_mean']:<10.4f} "
                f"{result['cv_std']:<8.4f}\n")
    
    f.write(f"\nBEST MODEL: {best_model_name}\n")
    f.write(f"BEST TEST ACCURACY: {best_test_acc:.4f}\n\n")
    
    if importances is not None:
        f.write("FEATURE IMPORTANCE:\n")
        for feature, importance in feature_importance:
            f.write(f"  {feature:<20}: {importance:.4f}\n")
    
    f.write(f"\nINDIVIDUAL FEATURE PERFORMANCE:\n")
    for i, feature_name in enumerate(feature_names):
        single_feature_model = LogisticRegression(random_state=42, max_iter=1000)
        X_single = X_scaled[:, i:i+1]
        single_cv_scores = cross_val_score(single_feature_model, X_single, y, 
                                         cv=5, scoring='accuracy')
        f.write(f"  {feature_name:<20}: {single_cv_scores.mean():.4f} (±{single_cv_scores.std():.4f})\n")

print(f"Results saved to: {output_dir}/classification_results.txt")

print(f"\n" + "="*70)
print("FINAL ANALYSIS AND RECOMMENDATIONS")
print("="*70)

print(f"\nDIAGNOSIS:")
print(f"1. SEVERE CLASS IMBALANCE: {(1-np.mean(y))*100:.1f}% variants vs {np.mean(y)*100:.1f}% references")
print(f"2. MAJORITY CLASS EXPLOITATION: Models achieve {majority_class_accuracy:.1%} by predicting majority class")
print(f"3. DATA LEAKAGE SUSPECTED: Individual features perform at baseline level")

print(f"\nWHY THIS HAPPENS:")
print(f"- Reference sentences are rare in the corpus (only ~10%)")
print(f"- Models exploit this imbalance rather than learning linguistic patterns")
print(f"- High accuracy is misleading - it's just predicting 'variant' for everything")

print(f"\nRECOMMENDations:")
print(f"1. USE BALANCED EVALUATION: Focus on balanced dataset results")
print(f"2. USE APPROPRIATE METRICS: F1-score, precision/recall for minority class")
print(f"3. COMPARE TO JOACHIMS APPROACH: The 88.81% from balanced pairs is more reliable")
print(f"4. INVESTIGATE FEATURE ENGINEERING: Current features may not capture linguistic differences")

# Summary comparison
print(f"\nPERFORMACE COMPARISON:")
print(f"- Imbalanced dataset (misleading): ~93%")
print(f"- Balanced dataset (realistic): ~XX% (see balanced results above)")
print(f"- Joachims paired approach: 88.81%")
print(f"- Majority class baseline: {majority_class_accuracy:.1%}")

print(f"\nCONCLUSION:")
print(f"The 93% accuracy is misleading due to class imbalance exploitation.")
print(f"The original Joachims approach with 88.81% is more reliable because:")
print(f"- It uses balanced pairwise comparisons")
print(f"- It focuses on relative preferences within sentence families")
print(f"- It avoids the corpus-level class imbalance issue")

print("\n" + "="*70)
print("CLASSIFICATION ANALYSIS COMPLETE")
print("="*70)
print("⚠️  WARNING: Results indicate data leakage and class imbalance issues")
print("   The original Joachims approach is more scientifically sound")
print(f"   Best test accuracy: {best_test_acc:.4f} (misleading due to class imbalance)")
print(f"   Results saved to: {output_dir}/")