#!/usr/bin/env python3
"""
Individual Sentence Classification: Reference vs Variant
With proper data balancing and no data leakage - FIXED
"""

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import GroupKFold
from sklearn.metrics import (accuracy_score, precision_recall_fscore_support, 
                            roc_auc_score, confusion_matrix, classification_report,
                            f1_score, balanced_accuracy_score)
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
import matplotlib.pyplot as plt
import seaborn as sns
from tqdm import tqdm
import warnings
from collections import defaultdict, Counter

warnings.filterwarnings('ignore')

# Set seeds
torch.manual_seed(42)
np.random.seed(42)

# Device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    print(f"GPU Memory: {torch.cuda.get_device_properties(0).total_memory / 1e9:.2f} GB")

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    def __init__(self, alpha=1, gamma=2):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        
    def forward(self, inputs, targets):
        ce_loss = nn.functional.binary_cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return torch.mean(focal_loss)

class SentenceDataset(Dataset):
    """Dataset for individual sentence classification"""
    def __init__(self, features, labels):
        self.features = torch.FloatTensor(features)
        self.labels = torch.FloatTensor(labels)
        
    def __len__(self):
        return len(self.labels)
    
    def __getitem__(self, idx):
        return self.features[idx], self.labels[idx]

class AttentionLayer(nn.Module):
    """Self-attention mechanism for feature importance"""
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.Tanh(),
            nn.Linear(input_dim // 2, 1)
        )
        
    def forward(self, x):
        # x shape: (batch_size, input_dim)
        # Create attention weights
        attn_weights = self.attention(x)
        attn_weights = torch.softmax(attn_weights, dim=0)
        
        # Apply attention
        weighted = x * attn_weights
        return weighted

class RobustSentenceClassifier(nn.Module):
    """
    Robust Neural Network for sentence classification
    with multiple regularization techniques
    """
    def __init__(self, input_dim, hidden_dims=[256, 128, 64], 
                 dropout_rates=[0.5, 0.4, 0.3], use_attention=True):
        super(RobustSentenceClassifier, self).__init__()
        
        self.use_attention = use_attention
        
        # Input batch normalization
        self.input_bn = nn.BatchNorm1d(input_dim)
        
        # Attention layer
        if use_attention:
            self.attention = AttentionLayer(input_dim)
        
        # Build deep network with residual connections
        self.layers = nn.ModuleList()
        self.dropouts = nn.ModuleList()
        self.batch_norms = nn.ModuleList()
        
        prev_dim = input_dim
        for i, (hidden_dim, dropout_rate) in enumerate(zip(hidden_dims, dropout_rates)):
            self.layers.append(nn.Linear(prev_dim, hidden_dim))
            self.batch_norms.append(nn.BatchNorm1d(hidden_dim))
            self.dropouts.append(nn.Dropout(dropout_rate))
            prev_dim = hidden_dim
        
        # Output layers with additional regularization
        self.output_layers = nn.Sequential(
            nn.Linear(prev_dim, prev_dim // 2),
            nn.BatchNorm1d(prev_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(prev_dim // 2, 1)
        )
        
        # Initialize weights
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.constant_(module.bias, 0)
    
    def forward(self, x):
        # Input normalization
        if x.size(0) > 1:
            x = self.input_bn(x)
        
        # Attention
        if self.use_attention:
            x = self.attention(x)
        
        # Forward through layers
        for i, (layer, bn, dropout) in enumerate(zip(self.layers, self.batch_norms, self.dropouts)):
            residual = x if i > 0 and x.size(1) == layer.out_features else None
            
            x = layer(x)
            if x.size(0) > 1:
                x = bn(x)
            x = torch.relu(x)
            x = dropout(x)
            
            # Residual connection if dimensions match
            if residual is not None and residual.size(1) == x.size(1):
                x = x + residual
        
        # Output
        return torch.sigmoid(self.output_layers(x))

def load_individual_sentence_data():
    """Load data for individual sentence classification"""
    print("Loading feature files...")
    
    # Load all features
    surprisal_df = pd.read_csv('hutb_trigram_surprisals.csv')
    dep_df = pd.read_csv('enhanced_dependency_detailed.csv')
    is_df = pd.read_csv('hutb_is_scores.csv')
    plm_df = pd.read_csv('hutb_plm_scores_robust.csv')
    cm_df = pd.read_csv('hutb_case_marker_scores.csv')
    
    # Fix PLM inversion
    plm_df['positional_lm_score'] = -plm_df['positional_lm_score']
    
    # Merge all features
    df = surprisal_df[['sentence_id', 'trigram_surprisal']].merge(
        dep_df[['sentence_id', 'dependency_length']], on='sentence_id', how='inner'
    )
    df = df.merge(is_df[['sentence_id', 'is_score']], on='sentence_id', how='left')
    df = df.merge(plm_df[['sentence_id', 'positional_lm_score']], on='sentence_id', how='inner')
    df = df.merge(cm_df[['sentence_id', 'case_marker_score']], on='sentence_id', how='inner')
    
    # Fill missing IS scores with 0
    df['is_score'] = df['is_score'].fillna(0)
    
    # Drop any remaining NaN values
    df = df.dropna()
    
    # Create labels: 1 for reference (*.0), 0 for variant
    df['is_reference'] = df['sentence_id'].apply(lambda x: 1 if x.endswith('.0') else 0)
    
    # Extract group ID for preventing data leakage
    df['group_id'] = df['sentence_id'].apply(lambda x: x.rsplit('.', 1)[0])
    
    return df

def create_advanced_features(df):
    """Create advanced features for better discrimination"""
    print("\nCreating advanced features...")
    
    # Basic features
    feature_cols = ['trigram_surprisal', 'dependency_length', 'is_score', 
                    'positional_lm_score', 'case_marker_score']
    
    # Ensure all feature columns are numeric
    for col in feature_cols:
        df[col] = pd.to_numeric(df[col], errors='coerce')
    
    # Drop rows with NaN in basic features
    df = df.dropna(subset=feature_cols)
    
    # Statistical features within groups
    for col in feature_cols:
        # Z-score within group
        df[f'{col}_zscore'] = df.groupby('group_id')[col].transform(
            lambda x: (x - x.mean()) / (x.std() + 1e-8)
        )
        
        # Rank within group
        df[f'{col}_rank'] = df.groupby('group_id')[col].rank(pct=True)
    
    # Interaction features
    df['surprisal_x_dep'] = df['trigram_surprisal'] * df['dependency_length']
    df['plm_x_cm'] = df['positional_lm_score'] * df['case_marker_score']
    df['is_x_dep'] = df['is_score'] * df['dependency_length']
    
    # Polynomial features
    for col in feature_cols:
        df[f'{col}_squared'] = df[col] ** 2
    
    # Get all numeric feature columns (excluding non-numeric columns)
    all_feature_cols = [col for col in df.columns if col not in 
                       ['sentence_id', 'is_reference', 'group_id'] and 
                       df[col].dtype in ['float64', 'int64']]
    
    # Final check for NaN values
    df = df.dropna(subset=all_feature_cols)
    
    return df, all_feature_cols

def balance_dataset(X_train, y_train, method='combined'):
    """Balance the dataset using various techniques"""
    print(f"\nBalancing dataset using {method} method...")
    print(f"Original distribution: {Counter(y_train)}")
    
    if len(np.unique(y_train)) < 2:
        print("Warning: Only one class present, skipping balancing")
        return X_train, y_train
    
    try:
        if method == 'smote':
            # SMOTE oversampling
            smote = SMOTE(random_state=42, k_neighbors=min(5, Counter(y_train)[1]-1))
            X_balanced, y_balanced = smote.fit_resample(X_train, y_train)
        
        elif method == 'undersample':
            # Random undersampling
            rus = RandomUnderSampler(random_state=42)
            X_balanced, y_balanced = rus.fit_resample(X_train, y_train)
        
        elif method == 'combined':
            # Combination: First oversample minority, then undersample majority
            minority_count = Counter(y_train)[1]
            if minority_count > 2:
                smote = SMOTE(random_state=42, k_neighbors=min(3, minority_count-1), 
                             sampling_strategy=0.7)
                X_temp, y_temp = smote.fit_resample(X_train, y_train)
                
                rus = RandomUnderSampler(random_state=42, sampling_strategy=0.8)
                X_balanced, y_balanced = rus.fit_resample(X_temp, y_temp)
            else:
                # If too few minority samples, just use undersampling
                rus = RandomUnderSampler(random_state=42)
                X_balanced, y_balanced = rus.fit_resample(X_train, y_train)
        
        else:
            X_balanced, y_balanced = X_train, y_train
            
    except Exception as e:
        print(f"Balancing failed: {e}")
        X_balanced, y_balanced = X_train, y_train
    
    print(f"Balanced distribution: {Counter(y_balanced)}")
    return X_balanced, y_balanced

def train_with_group_cv(df, feature_cols, n_splits=5):
    """Train with GroupKFold to prevent data leakage"""
    print(f"\nTraining with {n_splits}-fold GroupKFold cross-validation...")
    
    # Prepare data
    X = df[feature_cols].values
    y = df['is_reference'].values
    groups = df['group_id'].values
    
    # Ensure data is float
    X = X.astype(np.float32)
    y = y.astype(np.float32)
    
    # Print class distribution
    print(f"Overall class distribution: {Counter(y)}")
    print(f"Class ratio: {np.mean(y):.3f}")
    
    # Scale features
    scaler = StandardScaler()
    
    # Group K-Fold
    gkf = GroupKFold(n_splits=n_splits)
    
    # Store results
    cv_results = {
        'accuracy': [], 'balanced_accuracy': [], 'precision': [], 
        'recall': [], 'f1': [], 'auc': []
    }
    
    for fold, (train_idx, test_idx) in enumerate(gkf.split(X, y, groups)):
        print(f"\n--- Fold {fold+1}/{n_splits} ---")
        
        # Split data
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        groups_train, groups_test = groups[train_idx], groups[test_idx]
        
        # Verify no group overlap
        assert len(set(groups_train).intersection(set(groups_test))) == 0
        
        print(f"Train: {len(X_train)} samples, Test: {len(X_test)} samples")
        print(f"Train distribution: {Counter(y_train)}")
        print(f"Test distribution: {Counter(y_test)}")
        
        # Check if we have both classes in train set
        if len(np.unique(y_train)) < 2:
            print("Warning: Only one class in training set, skipping fold")
            continue
        
        # Scale features
        X_train_scaled = scaler.fit_transform(X_train)
        X_test_scaled = scaler.transform(X_test)
        
        # Balance training data
        X_train_balanced, y_train_balanced = balance_dataset(
            X_train_scaled, y_train, method='combined'
        )
        
        # Split for validation
        val_size = int(0.15 * len(X_train_balanced))
        X_val = X_train_balanced[:val_size]
        y_val = y_train_balanced[:val_size]
        X_train_final = X_train_balanced[val_size:]
        y_train_final = y_train_balanced[val_size:]
        
        # Calculate class weights for loss function
        class_counts = Counter(y_train_final)
        if len(class_counts) < 2:
            print("Warning: Only one class after balancing, using equal weights")
            class_weight_0 = 1.0
            class_weight_1 = 1.0
        else:
            total = sum(class_counts.values())
            class_weight_0 = total / (2 * max(class_counts[0], 1))
            class_weight_1 = total / (2 * max(class_counts[1], 1))
        
        # Create data loaders
        train_dataset = SentenceDataset(X_train_final, y_train_final)
        val_dataset = SentenceDataset(X_val, y_val)
        test_dataset = SentenceDataset(X_test_scaled, y_test)
        
        # Create weighted sampler for training
        sample_weights = [class_weight_1 if label == 1 else class_weight_0 
                         for label in y_train_final]
        sampler = WeightedRandomSampler(sample_weights, len(sample_weights))
        
        train_loader = DataLoader(train_dataset, batch_size=32, sampler=sampler)
        val_loader = DataLoader(val_dataset, batch_size=32, shuffle=False)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)
        
        # Create model
        model = RobustSentenceClassifier(
            input_dim=X.shape[1],
            hidden_dims=[128, 64, 32],
            dropout_rates=[0.5, 0.4, 0.3],
            use_attention=True
        ).to(device)
        
        # Loss and optimizer
        criterion = FocalLoss(alpha=2, gamma=2)  # Focal loss for imbalance
        optimizer = optim.AdamW(model.parameters(), lr=0.001, weight_decay=1e-4)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', patience=5)
        
        # Training loop with early stopping
        best_val_f1 = 0
        patience_counter = 0
        patience = 10
        best_model_state = None
        
        for epoch in range(50):
            # Training
            model.train()
            train_loss = 0
            train_pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
            
            for features, labels in train_pbar:
                features, labels = features.to(device), labels.to(device)
                
                optimizer.zero_grad()
                outputs = model(features).squeeze()
                loss = criterion(outputs, labels)
                
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                
                train_loss += loss.item()
                train_pbar.set_postfix({'loss': f'{loss.item():.4f}'})
            
            # Validation
            model.eval()
            val_predictions = []
            val_labels = []
            val_loss = 0
            
            with torch.no_grad():
                for features, labels in val_loader:
                    features, labels = features.to(device), labels.to(device)
                    outputs = model(features).squeeze()
                    loss = criterion(outputs, labels)
                    val_loss += loss.item()
                    
                    predictions = (outputs > 0.5).float()
                    val_predictions.extend(predictions.cpu().numpy())
                    val_labels.extend(labels.cpu().numpy())
            
            # Calculate validation metrics
            if len(np.unique(val_labels)) > 1:
                val_f1 = f1_score(val_labels, val_predictions)
            else:
                val_f1 = 0
                
            avg_val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
            
            # Learning rate scheduling
            scheduler.step(avg_val_loss)
            
            # Early stopping based on F1 score
            if val_f1 > best_val_f1:
                best_val_f1 = val_f1
                patience_counter = 0
                best_model_state = model.state_dict().copy()
            else:
                patience_counter += 1
                
            if patience_counter >= patience:
                break
        
        # Load best model
        if best_model_state is not None:
            model.load_state_dict(best_model_state)
        
        # Evaluate on test set
        model.eval()
        test_predictions = []
        test_probs = []
        test_labels = []
        
        with torch.no_grad():
            for features, labels in tqdm(test_loader, desc='Evaluating'):
                features = features.to(device)
                outputs = model(features).squeeze()
                
                predictions = (outputs > 0.5).float()
                test_predictions.extend(predictions.cpu().numpy())
                test_probs.extend(outputs.cpu().numpy())
                test_labels.extend(labels.cpu().numpy())
        
        # Calculate metrics
        test_predictions = np.array(test_predictions)
        test_probs = np.array(test_probs)
        test_labels = np.array(test_labels)
        
        # Handle case where test set might have only one class
        if len(np.unique(test_labels)) < 2:
            print("Warning: Test set has only one class")
            accuracy = accuracy_score(test_labels, test_predictions)
            balanced_acc = accuracy
            precision = recall = f1 = 0
            auc = 0.5
        else:
            accuracy = accuracy_score(test_labels, test_predictions)
            balanced_acc = balanced_accuracy_score(test_labels, test_predictions)
            precision, recall, f1, _ = precision_recall_fscore_support(
                test_labels, test_predictions, average='binary', zero_division=0
            )
            try:
                auc = roc_auc_score(test_labels, test_probs)
            except:
                auc = 0.5
        
        cv_results['accuracy'].append(accuracy)
        cv_results['balanced_accuracy'].append(balanced_acc)
        cv_results['precision'].append(precision)
        cv_results['recall'].append(recall)
        cv_results['f1'].append(f1)
        cv_results['auc'].append(auc)
        
        print(f"\nFold {fold+1} Results:")
        print(f"Accuracy: {accuracy:.4f}")
        print(f"Balanced Accuracy: {balanced_acc:.4f}")
        print(f"Precision: {precision:.4f}")
        print(f"Recall: {recall:.4f}")
        print(f"F1-Score: {f1:.4f}")
        print(f"AUC-ROC: {auc:.4f}")
        
        # Confusion matrix for this fold
        if len(np.unique(test_labels)) > 1:
            cm = confusion_matrix(test_labels, test_predictions)
            print(f"\nConfusion Matrix:")
            print(f"TN: {cm[0,0]}, FP: {cm[0,1]}")
            print(f"FN: {cm[1,0]}, TP: {cm[1,1]}")
    
    return cv_results

def main():
    """Main pipeline for individual sentence classification"""
    print("="*70)
    print("INDIVIDUAL SENTENCE CLASSIFICATION: REFERENCE vs VARIANT")
    print("="*70)
    
    # Load data
    df = load_individual_sentence_data()
    print(f"\nLoaded {len(df)} sentences")
    print(f"References: {sum(df['is_reference'])}")
    print(f"Variants: {sum(1 - df['is_reference'])}")
    print(f"Unique groups: {df['group_id'].nunique()}")
    
    # Create advanced features
    df, feature_cols = create_advanced_features(df)
    print(f"\nCreated {len(feature_cols)} features")
    print("Features:", feature_cols[:10], "...")  # Show first 10 features
    
    # Final check
    print(f"Final dataset size: {len(df)} sentences")
    
    # Verify data types
    print("\nVerifying data types...")
    for col in feature_cols:
        if df[col].dtype not in ['float64', 'int64', 'float32', 'int32']:
            print(f"Warning: {col} has dtype {df[col].dtype}")
    
    # Train with cross-validation
    cv_results = train_with_group_cv(df, feature_cols, n_splits=5)
    
    # Print final results
    print("\n" + "="*70)
    print("FINAL CROSS-VALIDATION RESULTS")
    print("="*70)
    
    for metric, values in cv_results.items():
        if len(values) > 0:
            mean_val = np.mean(values)
            std_val = np.std(values)
            print(f"{metric.capitalize():20s}: {mean_val:.4f} ± {std_val:.4f}")
        else:
            print(f"{metric.capitalize():20s}: No valid results")
    
    # Plot results if we have valid results
    if any(len(v) > 0 for v in cv_results.values()):
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        metrics = list(cv_results.keys())
        
        for idx, (metric, ax) in enumerate(zip(metrics, axes.flat)):
            values = cv_results[metric]
            if len(values) > 0:
                ax.bar(range(1, len(values)+1), values)
                ax.axhline(y=np.mean(values), color='r', linestyle='--', 
                           label=f'Mean: {np.mean(values):.3f}')
                ax.set_xlabel('Fold')
                ax.set_ylabel(metric.capitalize())
                ax.set_title(f'{metric.capitalize()} by Fold')
                ax.legend()
                ax.set_ylim(0, 1)
        
        plt.tight_layout()
        plt.show()

if __name__ == "__main__":
    main()