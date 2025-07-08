#!/usr/bin/env python3
"""
Logistic Regression Model using Trigram Surprisal and Dependency Length
Replicating Ranjan et al. (2022) with combined predictors
Expected accuracy: ~91.42% for combined model (as reported in paper)
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import KFold
from sklearn.metrics import accuracy_score
from tqdm import tqdm
import argparse
import json
from collections import defaultdict

class CombinedModel:
    def __init__(self, surprisal_file, dependency_file):
        """
        Initialize model with trigram surprisal scores and dependency lengths
        
        Args:
            surprisal_file: Path to CSV with trigram surprisal scores
            dependency_file: Path to CSV with dependency length scores
        """
        print(f"Loading surprisal scores from {surprisal_file}...")
        self.surprisal_df = pd.read_csv(surprisal_file)
        
        print(f"Loading dependency lengths from {dependency_file}...")
        self.dependency_df = pd.read_csv(dependency_file)
        
        # Create lookup dictionaries for fast access
        self.surprisal_lookup = {}
        for _, row in self.surprisal_df.iterrows():
            self.surprisal_lookup[row['sentence_id']] = row['trigram_surprisal']
        
        self.dependency_lookup = {}
        for _, row in self.dependency_df.iterrows():
            self.dependency_lookup[row['sentence_id']] = row['dependency_length']
        
        print(f"✓ Loaded {len(self.surprisal_lookup)} surprisal scores")
        print(f"✓ Loaded {len(self.dependency_lookup)} dependency lengths")
        
        # Initialize logistic regression
        self.lr_model = LogisticRegression(random_state=42, max_iter=1000)
        
    def load_reference_variant_pairs(self, pairs_file):
        """
        Load reference-variant pairs
        
        Expected format: JSON or CSV with reference and variant sentence IDs
        For now, we'll simulate based on naming convention in your data
        """
        # Parse sentence IDs to find reference-variant groups
        sentence_groups = defaultdict(list)
        
        # Use intersection of both lookups to ensure we have both features
        common_ids = set(self.surprisal_lookup.keys()) & set(self.dependency_lookup.keys())
        
        for sent_id in common_ids:
            # Extract base ID (without variant number)
            # Format: file-name__sentnum.variant
            parts = sent_id.rsplit('.', 1)  # Split from right to handle filenames with dots
            if len(parts) == 2:
                base_id = parts[0]
                try:
                    variant_num = int(parts[1])
                    sentence_groups[base_id].append((variant_num, sent_id))
                except ValueError:
                    continue
        
        # Create reference-variant pairs
        pairs = []
        for base_id, variants in sentence_groups.items():
            if len(variants) > 1:
                # Sort by variant number
                variants.sort(key=lambda x: x[0])
                
                # First variant (0) is reference
                reference_id = None
                variant_ids = []
                
                for var_num, sent_id in variants:
                    if var_num == 0:
                        reference_id = sent_id
                    else:
                        variant_ids.append(sent_id)
                
                # Create pairs
                if reference_id and variant_ids:
                    for var_id in variant_ids:
                        pairs.append((reference_id, var_id))
        
        print(f"✓ Created {len(pairs)} reference-variant pairs")
        return pairs
    
    def prepare_training_data(self, pairs):
        """
        Prepare training data using Joachims transformation
        
        Args:
            pairs: List of (reference_id, variant_id) tuples
            
        Returns:
            X: Feature matrix (n_samples, 2) - [trigram surprisal diff, dependency length diff]
            y: Labels (1 for reference > variant, 0 otherwise)
        """
        X = []
        y = []
        
        print("Preparing training data...")
        for ref_id, var_id in tqdm(pairs):
            # Get surprisal scores
            ref_surprisal = self.surprisal_lookup.get(ref_id, None)
            var_surprisal = self.surprisal_lookup.get(var_id, None)
            
            # Get dependency lengths
            ref_dep_len = self.dependency_lookup.get(ref_id, None)
            var_dep_len = self.dependency_lookup.get(var_id, None)
            
            if ref_surprisal is None or var_surprisal is None or ref_dep_len is None or var_dep_len is None:
                continue
            
            # Calculate differences
            δ_surprisal = var_surprisal - ref_surprisal  # Note: paper uses var - ref
            δ_dep_length = var_dep_len - ref_dep_len
            
            # Joachims transformation:
            # Create two data points for each pair
            
            # 1. Variant-Reference (label = 0)
            X.append([δ_surprisal, δ_dep_length])
            y.append(0)
            
            # 2. Reference-Variant (label = 1)
            X.append([-δ_surprisal, -δ_dep_length])
            y.append(1)
        
        X = np.array(X)
        y = np.array(y)
        
        print(f"✓ Created {len(X)} training samples from {len(pairs)} pairs")
        print(f"  Positive samples (ref chosen): {sum(y)}")
        print(f"  Negative samples (var chosen): {len(y) - sum(y)}")
        
        return X, y
    
    def train(self, X, y):
        """Train the model"""
        print("\nTraining logistic regression model...")
        self.lr_model.fit(X, y)
        
        # Print coefficients (should match paper's Table 3)
        print(f"\nModel coefficients (cf. Table 3 in paper):")
        print(f"  Trigram surprisal: {self.lr_model.coef_[0][0]:.6f}")
        print(f"  Dependency length: {self.lr_model.coef_[0][1]:.6f}")
        print(f"  Intercept: {self.lr_model.intercept_[0]:.6f}")
        
        # All coefficients should be negative (lower values → higher probability)
        print("\nCoefficient signs:")
        if self.lr_model.coef_[0][0] < 0:
            print("  ✓ Trigram surprisal: negative (correct)")
        else:
            print("  ✗ Trigram surprisal: positive (unexpected)")
            
        if self.lr_model.coef_[0][1] < 0:
            print("  ✓ Dependency length: negative (correct)")
        else:
            print("  ✗ Dependency length: positive (unexpected)")
    
    def evaluate_with_cv(self, X, y, n_splits=10):
        """
        Evaluate using k-fold cross-validation
        
        Returns:
            mean_accuracy: Average accuracy across folds
        """
        print(f"\nPerforming {n_splits}-fold cross-validation...")
        
        kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)
        accuracies = []
        
        for fold, (train_idx, test_idx) in enumerate(kf.split(X)):
            X_train, X_test = X[train_idx], X[test_idx]
            y_train, y_test = y[train_idx], y[test_idx]
            
            # Train on fold
            temp_model = LogisticRegression(random_state=42, max_iter=1000)
            temp_model.fit(X_train, y_train)
            
            # Test
            y_pred = temp_model.predict(X_test)
            acc = accuracy_score(y_test, y_pred)
            accuracies.append(acc)
            
            if fold < 3:  # Print details for first 3 folds
                print(f"  Fold {fold+1}: {acc:.4f} ({acc*100:.2f}%)")
        
        mean_acc = np.mean(accuracies)
        std_acc = np.std(accuracies)
        
        print(f"\nCross-validation results:")
        print(f"  Mean accuracy: {mean_acc:.4f} ({mean_acc*100:.2f}%)")
        print(f"  Std deviation: {std_acc:.4f}")
        print(f"  Min/Max: [{min(accuracies):.4f}, {max(accuracies):.4f}]")
        
        return mean_acc
    
    def test_individual_predictors(self, X, y):
        """Test each predictor individually to compare with paper"""
        print("\n" + "="*50)
        print("INDIVIDUAL PREDICTOR PERFORMANCE")
        print("="*50)
        
        # Trigram surprisal only
        X_surprisal = X[:, 0:1]  # First column only
        kf = KFold(n_splits=10, shuffle=True, random_state=42)
        surprisal_accs = []
        
        for train_idx, test_idx in kf.split(X_surprisal):
            temp_model = LogisticRegression(random_state=42, max_iter=1000)
            temp_model.fit(X_surprisal[train_idx], y[train_idx])
            acc = accuracy_score(y[test_idx], temp_model.predict(X_surprisal[test_idx]))
            surprisal_accs.append(acc)
        
        print(f"Trigram surprisal only: {np.mean(surprisal_accs)*100:.2f}%")
        print(f"  (Paper: 91.01%)")
        
        # Dependency length only
        X_deplen = X[:, 1:2]  # Second column only
        deplen_accs = []
        
        for train_idx, test_idx in kf.split(X_deplen):
            temp_model = LogisticRegression(random_state=42, max_iter=1000)
            temp_model.fit(X_deplen[train_idx], y[train_idx])
            acc = accuracy_score(y[test_idx], temp_model.predict(X_deplen[test_idx]))
            deplen_accs.append(acc)
        
        print(f"Dependency length only: {np.mean(deplen_accs)*100:.2f}%")
        print(f"  (Paper: 60.04%)")
        
        # Combined (already calculated)
        combined_acc = self.evaluate_with_cv(X, y)
        print(f"Combined model: {combined_acc*100:.2f}%")
        print(f"  (Paper: 91.42%)")
        
        return np.mean(surprisal_accs), np.mean(deplen_accs), combined_acc
    
    def analyze_feature_contributions(self, X, y, n_samples=20):
        """Analyze how each feature contributes to predictions"""
        print(f"\n\nAnalyzing feature contributions...")
        
        # Get predictions and probabilities
        y_pred = self.lr_model.predict(X)
        y_proba = self.lr_model.predict_proba(X)
        
        # Find interesting cases
        correct_preds = X[y_pred == y]
        wrong_preds = X[y_pred != y]
        
        print(f"\nCorrect predictions: {len(correct_preds)}/{len(X)} ({len(correct_preds)/len(X)*100:.1f}%)")
        
        # Analyze feature distributions
        print("\nFeature statistics (for correct predictions):")
        if len(correct_preds) > 0:
            print(f"  δ_surprisal: mean={correct_preds[:, 0].mean():.2f}, std={correct_preds[:, 0].std():.2f}")
            print(f"  δ_dep_length: mean={correct_preds[:, 1].mean():.2f}, std={correct_preds[:, 1].std():.2f}")


def main():
    parser = argparse.ArgumentParser(
        description="Test combined trigram surprisal + dependency length model"
    )
    parser.add_argument("--surprisal", default="hutb_trigram_surprisals.csv",
                       help="Path to trigram surprisal CSV")
    parser.add_argument("--dependency", default="enhanced_dependency_detailed.csv",
                       help="Path to dependency length CSV")
    parser.add_argument("--pairs", default=None,
                       help="Path to reference-variant pairs (optional)")
    parser.add_argument("--cv-folds", type=int, default=10,
                       help="Number of cross-validation folds")
    parser.add_argument("--test-individual", action="store_true",
                       help="Test individual predictors")
    parser.add_argument("--analyze", action="store_true",
                       help="Analyze feature contributions")
    
    args = parser.parse_args()
    
    # Initialize model
    model = CombinedModel(args.surprisal, args.dependency)
    
    # Load or generate pairs
    if args.pairs:
        # Load from file if provided
        print(f"Loading pairs from {args.pairs}...")
        # Implement loading logic based on your format
        pairs = []  # Placeholder
    else:
        # Generate from sentence IDs
        pairs = model.load_reference_variant_pairs(None)
    
    if not pairs:
        print("\n❌ No reference-variant pairs found!")
        return
    
    # Prepare data
    X, y = model.prepare_training_data(pairs)
    
    if len(X) == 0:
        print("\n❌ No valid training data created!")
        return
    
    # Train model
    model.train(X, y)
    
    # Test individual predictors if requested
    if args.test_individual:
        surp_acc, dep_acc, comb_acc = model.test_individual_predictors(X, y)
    else:
        # Just evaluate combined model
        accuracy = model.evaluate_with_cv(X, y, n_splits=args.cv_folds)
    
    # Analyze feature contributions if requested
    if args.analyze:
        model.analyze_feature_contributions(X, y)
    
    # Summary comparison with paper (Table 4a)
    print("\n" + "="*50)
    print("COMPARISON WITH PAPER (Table 4a)")
    print("="*50)
    print("Predictor               | Paper  | Ours")
    print("-----------------------|--------|--------")
    print("Dependency length      | 60.04% | ?")
    print("Trigram surprisal      | 91.01% | ?")
    print("Combined (All)         | 91.42% | ?")
    print("\nRun with --test-individual to see all results")


if __name__ == "__main__":
    main()