"""
CAFA 6 Protein Function Prediction - Data Preprocessing
========================================================
This script handles loading and preparing the CAFA 6 dataset
"""

import pandas as pd
import numpy as np
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load Training Data
# ============================================================================

def load_protein_sequences(fasta_file):
    """
    Load protein sequences from FASTA format file
    
    Args:
        fasta_file: Path to FASTA file
        
    Returns:
        Dictionary with protein_id as key and sequence as value
    """
    sequences = {}
    current_id = None
    current_seq = []
    
    with open(fasta_file, 'r') as f:
        for line in f:
            line = line.strip()
            
            # Header line starts with '>'
            if line.startswith('>'):
                # Save previous sequence if exists
                if current_id:
                    sequences[current_id] = ''.join(current_seq)
                
                # Extract protein ID (remove '>' and take first part)
                current_id = line[1:].split()[0]
                current_seq = []
            else:
                # Sequence line
                current_seq.append(line)
        
        # Don't forget the last sequence
        if current_id:
            sequences[current_id] = ''.join(current_seq)
    
    print(f" Loaded {len(sequences)} protein sequences")
    return sequences


def load_go_annotations(train_terms_file):
    """
    Load GO term annotations for training proteins
    
    Args:
        train_terms_file: Path to train_terms.tsv
        
    Returns:
        DataFrame with columns: EntryID, term, aspect
    """
    # Load TSV file
    df = pd.read_csv(train_terms_file, sep='\t')
    
    print(f" Loaded {len(df)} GO annotations")
    print(f"  - Unique proteins: {df['EntryID'].nunique()}")
    print(f"  - Unique GO terms: {df['term'].nunique()}")
    print(f"\nGO term distribution by aspect:")
    print(df['aspect'].value_counts())
    
    return df


def load_taxonomy_data(train_taxonomy_file):
    """
    Load taxonomic information for proteins
    
    Args:
        train_taxonomy_file: Path to train_taxonomy.tsv
        
    Returns:
        DataFrame with taxonomic info
    """
    df = pd.read_csv(train_taxonomy_file, sep='\t')
    print(f" Loaded taxonomy data for {len(df)} proteins")
    return df


# ============================================================================
# STEP 2: Create Training Dataset
# ============================================================================

def create_multilabel_dataset(sequences_dict, go_annotations_df, top_n_terms=1000):
    """
    Create multi-label dataset for training
    
    Args:
        sequences_dict: Dictionary of protein sequences
        go_annotations_df: DataFrame with GO annotations
        top_n_terms: Number of most common GO terms to use
        
    Returns:
        X: List of sequences
        y: Binary matrix (n_proteins x n_go_terms)
        go_terms: List of GO term IDs
    """
    # Get most common GO terms
    term_counts = go_annotations_df['term'].value_counts()
    top_terms = term_counts.head(top_n_terms).index.tolist()
    
    print(f"\n Using top {top_n_terms} GO terms")
    print(f"Coverage: {term_counts.head(top_n_terms).sum() / term_counts.sum() * 100:.1f}%")
    
    # Filter annotations to only include top terms
    filtered_df = go_annotations_df[go_annotations_df['term'].isin(top_terms)]
    
    # Get unique proteins that have annotations in top terms
    protein_ids = filtered_df['EntryID'].unique()
    
    # Create binary matrix
    X = []  # Sequences
    y = []  # Binary labels
    
    for protein_id in protein_ids:
        # Skip if sequence not available
        if protein_id not in sequences_dict:
            continue
            
        # Get protein sequence
        X.append(sequences_dict[protein_id])
        
        # Get GO terms for this protein
        protein_terms = filtered_df[filtered_df['EntryID'] == protein_id]['term'].values
        
        # Create binary vector
        binary_vector = [1 if term in protein_terms else 0 for term in top_terms]
        y.append(binary_vector)
    
    y = np.array(y)
    
    print(f"\n✓ Dataset created:")
    print(f"  - Proteins: {len(X)}")
    print(f"  - GO terms: {len(top_terms)}")
    print(f"  - Average labels per protein: {y.sum(axis=1).mean():.1f}")
    print(f"  - Label sparsity: {(1 - y.mean()) * 100:.1f}%")
    
    return X, y, top_terms


# ============================================================================
# STEP 3: Sequence Statistics
# ============================================================================

def analyze_sequences(sequences_dict):
    """
    Analyze protein sequence statistics
    
    Args:
        sequences_dict: Dictionary of sequences
    """
    lengths = [len(seq) for seq in sequences_dict.values()]
    
    print("\n Sequence Length Statistics:")
    print(f"  - Mean: {np.mean(lengths):.0f}")
    print(f"  - Median: {np.median(lengths):.0f}")
    print(f"  - Min: {np.min(lengths)}")
    print(f"  - Max: {np.max(lengths)}")
    print(f"  - Std: {np.std(lengths):.0f}")
    
    # Amino acid composition (example with first sequence)
    first_seq = list(sequences_dict.values())[0]
    amino_acids = set(first_seq)
    print(f"\n Amino Acids found: {sorted(amino_acids)}")
    print(f"   Total unique: {len(amino_acids)} (standard is 20)")


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Define paths (adjust these to your actual file paths)
    DATA_DIR = Path("data")
    
    # Example paths - you need to download CAFA 6 data from Kaggle
    TRAIN_SEQUENCES = DATA_DIR / "Train" / "train_sequences.fasta"
    TRAIN_TERMS = DATA_DIR / "Train" / "train_terms.tsv"
    TRAIN_TAXONOMY = DATA_DIR / "Train" / "train_taxonomy.tsv"
    
    # Check if files exist
    if not TRAIN_SEQUENCES.exists():
        print("  Files not found. Please download CAFA 6 data from Kaggle:")
        print("   https://www.kaggle.com/competitions/cafa-6-protein-function-prediction/data")
    else:
        # Load data
        print("=" * 70)
        print("LOADING CAFA 6 DATASET")
        print("=" * 70)
        
        sequences = load_protein_sequences(TRAIN_SEQUENCES)
        go_annotations = load_go_annotations(TRAIN_TERMS)
        taxonomy = load_taxonomy_data(TRAIN_TAXONOMY)
        
        # Analyze sequences
        analyze_sequences(sequences)
        
        # Create training dataset
        X, y, go_terms = create_multilabel_dataset(
            sequences, 
            go_annotations, 
            top_n_terms=1000
        )
        
        print("\n" + "=" * 70)
        print(" DATA PREPARATION COMPLETE")
        print("=" * 70)
        print(f"Ready for embedding extraction!")