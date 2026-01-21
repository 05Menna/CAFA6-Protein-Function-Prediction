"""
Extract ESM2 embeddings from training and test sequences
"""

import os
import sys
import pandas as pd
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from features.esm2_embeddings import (
    load_esm2_model,
    extract_embeddings_batch,
    extract_and_save_embeddings
)


def parse_fasta(fasta_file):
    """Parse FASTA file and return sequences and IDs"""
    sequences = []
    sequence_ids = []
    
    with open(fasta_file, 'r') as f:
        current_id = None
        current_seq = []
        
        for line in f:
            line = line.strip()
            if line.startswith('>'):
                # Save previous sequence
                if current_id is not None:
                    sequences.append(''.join(current_seq))
                    sequence_ids.append(current_id)
                
                # Start new sequence
                current_id = line[1:].split()[0]  # Get ID from header
                current_seq = []
            else:
                current_seq.append(line)
        
        # Save last sequence
        if current_id is not None:
            sequences.append(''.join(current_seq))
            sequence_ids.append(current_id)
    
    return sequences, sequence_ids


def main():
    # Setup paths
    data_dir = Path(__file__).parent.parent / "data"
    embeddings_dir = Path(__file__).parent.parent / "embeddings"
    embeddings_dir.mkdir(exist_ok=True)
    
    # Load model
    print("Loading ESM2 model...")
    model, tokenizer, device = load_esm2_model("facebook/esm2_t12_35M_UR50D")
    
    # Extract training embeddings
    print("\n" + "=" * 70)
    print("EXTRACTING TRAINING EMBEDDINGS")
    print("=" * 70)
    
    train_fasta = data_dir / "Train" / "train_sequences.fasta"
    if train_fasta.exists():
        print(f"\nParsing {train_fasta}...")
        train_sequences, train_ids = parse_fasta(str(train_fasta))
        print(f"✓ Loaded {len(train_sequences)} training sequences")
        
        # Extract embeddings
        output_file = embeddings_dir / "train_embeddings.npz"
        extract_and_save_embeddings(
            train_sequences,
            train_ids,
            model,
            tokenizer,
            device,
            str(output_file),
            batch_size=8,
            save_every=100
        )
    else:
        print(f"✗ Training file not found: {train_fasta}")
    
    # Extract test embeddings
    print("\n" + "=" * 70)
    print("EXTRACTING TEST EMBEDDINGS")
    print("=" * 70)
    
    test_fasta = data_dir / "Test" / "testsuperset.fasta"
    if test_fasta.exists():
        print(f"\nParsing {test_fasta}...")
        test_sequences, test_ids = parse_fasta(str(test_fasta))
        print(f"✓ Loaded {len(test_sequences)} test sequences")
        
        # Extract embeddings
        output_file = embeddings_dir / "test_embeddings.npz"
        extract_and_save_embeddings(
            test_sequences,
            test_ids,
            model,
            tokenizer,
            device,
            str(output_file),
            batch_size=8,
            save_every=100
        )
    else:
        print(f"✗ Test file not found: {test_fasta}")
    
    print("\n" + "=" * 70)
    print("EXTRACTION COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
