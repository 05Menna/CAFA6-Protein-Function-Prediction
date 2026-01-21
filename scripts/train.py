"""
CAFA 6 - Complete End-to-End Pipeline
======================================
This combines all steps: data loading → embeddings → training → prediction

Usage:
    1. Download CAFA 6 data from Kaggle
    2. Run this script
    3. Get predictions for test set
"""

import numpy as np
import pandas as pd
from pathlib import Path
import pickle
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# في scripts/train.py
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

# Now you can import from src
from src.data.data_loader import load_go_annotations, load_protein_sequences, load_taxonomy_data, create_multilabel_dataset
from src.features.esm2_embeddings import load_esm2_model
from src.models.binary_relevance import train_binary_relevance_xgboost
# ============================================================================
# CONFIGURATION
# ============================================================================

class Config:
    """Configuration for the pipeline"""
    
    # Paths
    DATA_DIR = Path("data")
    MODELS_DIR = Path("models")
    EMBEDDINGS_DIR = Path("embeddings")
    
    # Data files
    TRAIN_SEQUENCES = DATA_DIR / "Train" / "train_sequences.fasta"
    TRAIN_TERMS = DATA_DIR / "Train" / "train_terms.tsv"
    TRAIN_TAXONOMY = DATA_DIR / "Train" / "train_taxonomy.tsv"
    TEST_SEQUENCES = DATA_DIR / "Test" / "test_sequences.fasta"
    
    # Model settings
    ESM2_MODEL = "facebook/esm2_t12_35M_UR50D"  # Change to bigger for better quality
    TOP_N_GO_TERMS = 1000  # Number of GO terms to predict
    
    # Training settings
    VALIDATION_SPLIT = 0.2
    XGBOOST_N_ESTIMATORS = 100
    BATCH_SIZE = 8
    
    # Output
    SUBMISSION_FILE = "submission.tsv"
    
    def __init__(self):
        # Create directories
        self.MODELS_DIR.mkdir(exist_ok=True)
        self.EMBEDDINGS_DIR.mkdir(exist_ok=True)


# ============================================================================
# STEP 1: Data Loading and Preparation
# ============================================================================

def load_and_prepare_data(config):
    """
    Load all training data and prepare for modeling
    
    Returns:
        sequences_dict, go_annotations, taxonomy, X, y, go_terms
    """
    print("=" * 70)
    print("STEP 1: LOADING AND PREPARING DATA")
    print("=" * 70)
    
    # Load sequences (using function from previous script)
    print("\n Loading sequences...")
    sequences = load_protein_sequences(config.TRAIN_SEQUENCES)
    
    # Load GO annotations
    print("\n Loading GO annotations...")
    go_annotations = load_go_annotations(config.TRAIN_TERMS)
    
    # Load taxonomy
    print("\n Loading taxonomy...")
    taxonomy = load_taxonomy_data(config.TRAIN_TAXONOMY)
    
    # Create training dataset
    print("\n Creating training dataset...")
    X, y, go_terms = create_multilabel_dataset(
        sequences,
        go_annotations,
        top_n_terms=config.TOP_N_GO_TERMS
    )
    
    return sequences, go_annotations, taxonomy, X, y, go_terms


# ============================================================================
# STEP 2: Extract or Load Embeddings
# ============================================================================

def get_embeddings(sequences, config, force_recompute=False):
    """
    Extract ESM2 embeddings or load from cache
    
    Args:
        sequences: List of protein sequences
        config: Configuration object
        force_recompute: If True, recompute even if cache exists
        
    Returns:
        embeddings: numpy array of embeddings
    """
    embeddings_file = config.EMBEDDINGS_DIR / "train_embeddings.npz"
    
    # Check if embeddings already exist
    if embeddings_file.exists() and not force_recompute:
        print("\n Loading cached embeddings...")
        embeddings, _ = load_embeddings(embeddings_file)
        return embeddings
    
    print("=" * 70)
    print("STEP 2: EXTRACTING ESM2 EMBEDDINGS")
    print("=" * 70)
    
    # Load ESM2 model
    model, tokenizer, device = load_esm2_model(config.ESM2_MODEL)
    
    # Extract embeddings
    embeddings = extract_embeddings_batch(
        sequences,
        model,
        tokenizer,
        device,
        batch_size=config.BATCH_SIZE
    )
    
    # Save for future use
    print(f"\n Saving embeddings to {embeddings_file}")
    np.savez_compressed(
        embeddings_file,
        embeddings=embeddings,
        protein_ids=np.arange(len(sequences))
    )
    
    return embeddings


# ============================================================================
# STEP 3: Train Models
# ============================================================================

def train_models(X, y, go_terms, config):
    """
    Train multi-label classifier
    
    Returns:
        models: List of trained XGBoost models
        metrics: Validation metrics
    """
    print("=" * 70)
    print("STEP 3: TRAINING MODELS")
    print("=" * 70)
    
    # Split data
    from sklearn.model_selection import train_test_split
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, 
        test_size=config.VALIDATION_SPLIT,
        random_state=42
    )
    
    print(f"\n Data split:")
    print(f"   Training: {len(X_train)}")
    print(f"   Validation: {len(X_val)}")
    
    # Train Binary Relevance models
    models = train_binary_relevance_xgboost(
        X_train, y_train,
        X_val, y_val,
        n_estimators=config.XGBOOST_N_ESTIMATORS
    )
    
    # Evaluate
    y_pred, y_proba = predict_binary_relevance(models, X_val)
    metrics = evaluate_multilabel(y_val, y_pred)
    print_metrics(metrics, "Validation Metrics")
    
    # Save models
    models_file = config.MODELS_DIR / "xgboost_models.pkl"
    print(f"\n Saving models to {models_file}")
    with open(models_file, 'wb') as f:
        pickle.dump({'models': models, 'go_terms': go_terms}, f)
    
    return models, metrics


# ============================================================================
# STEP 4: Make Predictions on Test Set
# ============================================================================

def predict_test_set(config, models, go_terms):
    """
    Make predictions on test set and create submission file
    
    Args:
        config: Configuration
        models: Trained models
        go_terms: List of GO term IDs
        
    Returns:
        submission_df: DataFrame ready for submission
    """
    print("=" * 70)
    print("STEP 4: PREDICTING TEST SET")
    print("=" * 70)
    
    # Load test sequences
    print("\n Loading test sequences...")
    test_sequences = load_protein_sequences(config.TEST_SEQUENCES)
    test_ids = list(test_sequences.keys())
    test_seqs = list(test_sequences.values())
    
    # Extract embeddings for test set
    print("\n Extracting embeddings for test set...")
    model, tokenizer, device = load_esm2_model(config.ESM2_MODEL)
    X_test = extract_embeddings_batch(
        test_seqs,
        model,
        tokenizer,
        device,
        batch_size=config.BATCH_SIZE
    )
    
    # Make predictions
    print("\n Making predictions...")
    y_pred, y_proba = predict_binary_relevance(models, X_test)
    
    # Create submission file in CAFA format
    print("\n Creating submission file...")
    submission_rows = []
    
    for i, protein_id in enumerate(tqdm(test_ids, desc="Formatting submission")):
        # Get GO terms with probability > threshold
        predicted_indices = np.where(y_proba[i] > 0.01)[0]  # Low threshold to include more
        
        for idx in predicted_indices:
            go_term = go_terms[idx]
            probability = y_proba[i, idx]
            
            submission_rows.append({
                'EntryID': protein_id,
                'term': go_term,
                'probability': probability
            })
    
    submission_df = pd.DataFrame(submission_rows)
    
    # Sort by protein ID and probability
    submission_df = submission_df.sort_values(['EntryID', 'probability'], ascending=[True, False])
    
    # Save submission
    submission_df.to_csv(config.SUBMISSION_FILE, sep='\t', index=False)
    
    print(f"\n Submission saved to {config.SUBMISSION_FILE}")
    print(f"   Total predictions: {len(submission_df)}")
    print(f"   Unique proteins: {submission_df['EntryID'].nunique()}")
    print(f"   Avg predictions per protein: {len(submission_df) / submission_df['EntryID'].nunique():.1f}")
    
    return submission_df


# ============================================================================
# MAIN PIPELINE
# ============================================================================

def main():
    """
    Run the complete pipeline
    """
    print("\n" + "=" * 70)
    print("CAFA 6 PROTEIN FUNCTION PREDICTION - COMPLETE PIPELINE")
    print("=" * 70 + "\n")
    
    # Initialize configuration
    config = Config()
    
    # Check if data exists
    if not config.TRAIN_SEQUENCES.exists():
        print(" ERROR: CAFA 6 data not found!")
        print("\nPlease download the data from:")
        print("https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/data")
        print(f"\nAnd extract to: {config.DATA_DIR}")
        return
    
    # STEP 1: Load data
    sequences, go_annotations, taxonomy, X_seqs, y, go_terms = load_and_prepare_data(config)
    
    # STEP 2: Get embeddings
    X = get_embeddings(X_seqs, config, force_recompute=False)
    
    # STEP 3: Train models
    models, metrics = train_models(X, y, go_terms, config)
    
    # STEP 4: Predict test set
    submission = predict_test_set(config, models, go_terms)
    
    print("\n" + "=" * 70)
    print(" PIPELINE COMPLETE!")
    print("=" * 70)
    print(f"\nYour submission is ready: {config.SUBMISSION_FILE}")
    print("Upload it to Kaggle to see your score!")
    print("\nGood luck! ")


# ============================================================================
# RUN THE PIPELINE
# ============================================================================

if __name__ == "__main__":
    main()