"""
CAFA 6 - Multi-label Classification
====================================
Train XGBoost classifier to predict GO terms from protein embeddings

Multi-label means each protein can have multiple GO terms (labels)
"""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import f1_score, precision_score, recall_score
from sklearn.multioutput import MultiOutputClassifier
import xgboost as xgb
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Multi-label Classification Metrics
# ============================================================================

def evaluate_multilabel(y_true, y_pred):
    """
    Calculate evaluation metrics for multi-label classification
    
    Args:
        y_true: True binary labels (n_samples, n_labels)
        y_pred: Predicted binary labels (n_samples, n_labels)
        
    Returns:
        Dictionary with metrics
    """
    # Calculate metrics (average='samples' means per-protein average)
    metrics = {
        'f1_macro': f1_score(y_true, y_pred, average='macro', zero_division=0),
        'f1_micro': f1_score(y_true, y_pred, average='micro', zero_division=0),
        'f1_samples': f1_score(y_true, y_pred, average='samples', zero_division=0),
        'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
    }
    
    return metrics


def print_metrics(metrics, title="Metrics"):
    """Print metrics in a nice format"""
    print(f"\n{'=' * 50}")
    print(f"{title}")
    print(f"{'=' * 50}")
    for metric_name, value in metrics.items():
        print(f"{metric_name:15s}: {value:.4f}")
    print(f"{'=' * 50}")


# ============================================================================
# STEP 2: Simple Binary Relevance Approach
# ============================================================================

def train_binary_relevance_xgboost(X_train, y_train, X_val, y_val, n_estimators=100):
    """
    Train one XGBoost classifier per GO term (Binary Relevance)
    
    This is the simplest multi-label approach:
    - Train separate binary classifier for each label
    - Advantage: Simple, parallelizable
    - Disadvantage: Ignores label correlations
    
    Args:
        X_train: Training embeddings (n_samples, n_features)
        y_train: Training labels (n_samples, n_labels)
        X_val: Validation embeddings
        y_val: Validation labels
        n_estimators: Number of trees per classifier
        
    Returns:
        List of trained models (one per label)
    """
    n_labels = y_train.shape[1]
    models = []
    
    print(f"\n🌲 Training {n_labels} XGBoost classifiers...")
    print(f"   - Training samples: {len(X_train)}")
    print(f"   - Features: {X_train.shape[1]}")
    print(f"   - Trees per classifier: {n_estimators}")
    
    for i in tqdm(range(n_labels), desc="Training classifiers"):
        # Get labels for this GO term
        y_train_i = y_train[:, i]
        
        # Skip if all labels are 0 (no positive examples)
        if y_train_i.sum() == 0:
            models.append(None)
            continue
        
        # Train XGBoost classifier
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            objective='binary:logistic',
            eval_metric='logloss',
            use_label_encoder=False,
            random_state=42,
            n_jobs=-1  # Use all CPU cores
        )
        
        model.fit(
            X_train, y_train_i,
            eval_set=[(X_val, y_val[:, i])],
            verbose=False
        )
        
        models.append(model)
    
    print(f"✓ Training complete! {len([m for m in models if m is not None])} models trained")
    
    return models


def predict_binary_relevance(models, X, threshold=0.5):
    """
    Make predictions using trained Binary Relevance models
    
    Args:
        models: List of trained XGBoost models
        X: Input embeddings
        threshold: Probability threshold for positive prediction
        
    Returns:
        y_pred: Binary predictions (n_samples, n_labels)
        y_proba: Probability predictions (n_samples, n_labels)
    """
    n_samples = X.shape[0]
    n_labels = len(models)
    
    y_pred = np.zeros((n_samples, n_labels), dtype=int)
    y_proba = np.zeros((n_samples, n_labels), dtype=float)
    
    print(f"\n🔮 Making predictions...")
    
    for i in tqdm(range(n_labels), desc="Predicting"):
        if models[i] is None:
            continue
        
        # Get probability predictions
        proba = models[i].predict_proba(X)[:, 1]
        y_proba[:, i] = proba
        
        # Apply threshold
        y_pred[:, i] = (proba >= threshold).astype(int)
    
    return y_pred, y_proba


# ============================================================================
# STEP 3: Classifier Chain Approach (Advanced)
# ============================================================================

def train_classifier_chain(X_train, y_train, X_val, y_val, n_estimators=100):
    """
    Train Classifier Chain for multi-label classification
    
    Classifier Chain considers label dependencies:
    - Each classifier uses predictions from previous classifiers as features
    - Better than Binary Relevance but slower
    
    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_val: Validation embeddings
        y_val: Validation labels
        n_estimators: Trees per classifier
        
    Returns:
        List of trained models with feature indices
    """
    n_labels = y_train.shape[1]
    models = []
    
    print(f"\n🔗 Training Classifier Chain ({n_labels} classifiers)...")
    
    for i in tqdm(range(n_labels), desc="Chain training"):
        y_train_i = y_train[:, i]
        
        if y_train_i.sum() == 0:
            models.append(None)
            continue
        
        # Augment features with predictions from previous classifiers
        if i == 0:
            # First classifier: use only original features
            X_train_augmented = X_train
            X_val_augmented = X_val
        else:
            # Add previous predictions as features
            prev_train_preds = y_train[:, :i]
            prev_val_preds = y_val[:, :i]
            
            X_train_augmented = np.hstack([X_train, prev_train_preds])
            X_val_augmented = np.hstack([X_val, prev_val_preds])
        
        # Train classifier
        model = xgb.XGBClassifier(
            n_estimators=n_estimators,
            max_depth=6,
            learning_rate=0.1,
            random_state=42,
            n_jobs=-1
        )
        
        model.fit(
            X_train_augmented, y_train_i,
            eval_set=[(X_val_augmented, y_val[:, i])],
            verbose=False
        )
        
        models.append(model)
    
    print(f"✓ Classifier Chain trained!")
    
    return models


# ============================================================================
# STEP 4: Simple Baseline - K Nearest Neighbors (for comparison)
# ============================================================================

def knn_baseline(X_train, y_train, X_test, k=5):
    """
    Simple K-NN baseline for multi-label classification
    
    For each test protein:
    - Find K most similar training proteins
    - Aggregate their labels
    
    Args:
        X_train: Training embeddings
        y_train: Training labels
        X_test: Test embeddings
        k: Number of neighbors
        
    Returns:
        y_pred: Predicted labels
    """
    from sklearn.neighbors import NearestNeighbors
    
    print(f"\n👥 K-NN Baseline (k={k})...")
    
    # Find k nearest neighbors
    knn = NearestNeighbors(n_neighbors=k, metric='cosine')
    knn.fit(X_train)
    
    distances, indices = knn.kneighbors(X_test)
    
    # Aggregate neighbor labels
    y_pred = np.zeros((len(X_test), y_train.shape[1]))
    
    for i in range(len(X_test)):
        neighbor_labels = y_train[indices[i]]
        # Average neighbor labels (weighted by similarity)
        weights = 1 / (distances[i] + 1e-6)
        y_pred[i] = (neighbor_labels.T @ weights) / weights.sum()
    
    # Threshold at 0.5
    y_pred = (y_pred >= 0.5).astype(int)
    
    return y_pred


# ============================================================================
# STEP 5: Complete Training Pipeline
# ============================================================================

def train_and_evaluate_model(X, y, go_terms, test_size=0.2):
    """
    Complete pipeline: split data, train, evaluate
    
    Args:
        X: Protein embeddings
        y: GO term labels
        go_terms: List of GO term IDs
        test_size: Fraction for validation
        
    Returns:
        models, metrics
    """
    print("=" * 70)
    print("TRAINING PIPELINE")
    print("=" * 70)
    
    # Split data
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=test_size, random_state=42
    )
    
    print(f"\n📊 Data Split:")
    print(f"   Training: {len(X_train)} proteins")
    print(f"   Validation: {len(X_val)} proteins")
    print(f"   GO terms: {len(go_terms)}")
    
    # Train models
    models = train_binary_relevance_xgboost(
        X_train, y_train,
        X_val, y_val,
        n_estimators=100
    )
    
    # Make predictions
    y_pred, y_proba = predict_binary_relevance(models, X_val)
    
    # Evaluate
    metrics = evaluate_multilabel(y_val, y_pred)
    print_metrics(metrics, "Validation Results")
    
    # Compare with K-NN baseline
    print("\n🔄 Comparing with K-NN baseline...")
    y_pred_knn = knn_baseline(X_train, y_train, X_val, k=5)
    metrics_knn = evaluate_multilabel(y_val, y_pred_knn)
    print_metrics(metrics_knn, "K-NN Baseline")
    
    return models, metrics


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Simulate some data
    print("=" * 70)
    print("MULTI-LABEL CLASSIFICATION - DEMO")
    print("=" * 70)
    
    # Create dummy data
    np.random.seed(42)
    n_samples = 1000
    n_features = 480  # ESM2 embedding size
    n_labels = 100    # Number of GO terms
    
    X = np.random.randn(n_samples, n_features)
    y = (np.random.rand(n_samples, n_labels) > 0.95).astype(int)  # Sparse labels
    
    go_terms = [f"GO:{i:07d}" for i in range(n_labels)]
    
    print(f"\nDummy dataset:")
    print(f"  Samples: {n_samples}")
    print(f"  Features: {n_features}")
    print(f"  Labels: {n_labels}")
    print(f"  Avg labels per sample: {y.sum(axis=1).mean():.2f}")
    
    # Train and evaluate
    models, metrics = train_and_evaluate_model(X, y, go_terms)
    
    print("\n" + "=" * 70)
    print("✅ PIPELINE COMPLETE")
    print("=" * 70)