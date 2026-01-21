"""
CAFA 6 - ESM2 Embeddings Extraction
====================================
Extract protein embeddings using ESM2 (Evolutionary Scale Modeling)

ESM2 is a protein language model trained on 65 million protein sequences.
It converts amino acid sequences into meaningful numerical representations.
"""

import torch
import numpy as np
from transformers import AutoTokenizer, EsmModel
from tqdm import tqdm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# STEP 1: Load ESM2 Model
# ============================================================================

def load_esm2_model(model_name="facebook/esm2_t12_35M_UR50D"):
    """
    Load ESM2 model and tokenizer from HuggingFace
    
    Available models (sorted by size):
    - esm2_t6_8M_UR50D      (8M parameters)   - Fastest, lowest quality
    - esm2_t12_35M_UR50D    (35M parameters)  - Good balance ✓ RECOMMENDED
    - esm2_t30_150M_UR50D   (150M parameters) - Better quality
    - esm2_t33_650M_UR50D   (650M parameters) - High quality
    - esm2_t36_3B_UR50D     (3B parameters)   - Best quality, very slow
    
    Args:
        model_name: HuggingFace model identifier
        
    Returns:
        model, tokenizer
    """
    print(f" Loading {model_name}...")
    
    # Load tokenizer (converts sequences to tokens)
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    
    # Load model
    model = EsmModel.from_pretrained(model_name)
    
    # Move to GPU if available
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = model.to(device)
    model.eval()  # Set to evaluation mode
    
    print(f"✓ Model loaded on {device}")
    print(f"  Embedding dimension: {model.config.hidden_size}")
    
    return model, tokenizer, device


# ============================================================================
# STEP 2: Extract Embeddings for Single Sequence
# ============================================================================

def get_sequence_embedding(sequence, model, tokenizer, device, pooling='mean'):
    """
    Extract embedding for a single protein sequence
    
    Args:
        sequence: Amino acid sequence (string)
        model: ESM2 model
        tokenizer: ESM2 tokenizer
        device: torch device
        pooling: How to combine token embeddings ('mean', 'cls', 'max')
        
    Returns:
        Embedding vector (numpy array)
    """
    # Tokenize sequence (add special tokens)
    inputs = tokenizer(sequence, return_tensors="pt", truncation=True, max_length=1024)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    
    # Get embeddings (no gradient computation needed)
    with torch.no_grad():
        outputs = model(**inputs)
    
    # Get hidden states (embeddings for each amino acid)
    # Shape: (1, sequence_length, embedding_dim)
    hidden_states = outputs.last_hidden_state
    
    # Pool embeddings into a single vector
    if pooling == 'mean':
        # Average all token embeddings (excluding padding)
        mask = inputs['attention_mask'].unsqueeze(-1)
        embedding = (hidden_states * mask).sum(dim=1) / mask.sum(dim=1)
    elif pooling == 'cls':
        # Use the [CLS] token embedding (first token)
        embedding = hidden_states[:, 0, :]
    elif pooling == 'max':
        # Max pooling over all tokens
        embedding = torch.max(hidden_states, dim=1)[0]
    
    # Convert to numpy and remove batch dimension
    embedding = embedding.cpu().numpy()[0]
    
    return embedding


# ============================================================================
# STEP 3: Batch Processing for Multiple Sequences
# ============================================================================

def extract_embeddings_batch(sequences, model, tokenizer, device, 
                             batch_size=8, pooling='mean'):
    """
    Extract embeddings for multiple protein sequences
    
    Args:
        sequences: List of amino acid sequences
        model: ESM2 model
        tokenizer: ESM2 tokenizer
        device: torch device
        batch_size: Number of sequences to process at once
        pooling: Pooling strategy
        
    Returns:
        embeddings: numpy array of shape (n_sequences, embedding_dim)
    """
    all_embeddings = []
    
    print(f"\n Extracting embeddings for {len(sequences)} sequences...")
    print(f"   Batch size: {batch_size}")
    print(f"   Pooling: {pooling}")
    
    # Process in batches
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_sequences = sequences[i:i + batch_size]
        
        # Process each sequence in batch
        batch_embeddings = []
        for seq in batch_sequences:
            emb = get_sequence_embedding(seq, model, tokenizer, device, pooling)
            batch_embeddings.append(emb)
        
        all_embeddings.extend(batch_embeddings)
    
    # Convert to numpy array
    embeddings = np.array(all_embeddings)
    
    print(f" Embeddings shape: {embeddings.shape}")
    
    return embeddings


# ============================================================================
# STEP 4: Memory-Efficient Processing for Large Datasets
# ============================================================================

def extract_and_save_embeddings(sequences, protein_ids, model, tokenizer, device,
                                output_file, batch_size=8, save_every=1000):
    """
    Extract embeddings and save incrementally to avoid memory issues
    
    Args:
        sequences: List of sequences
        protein_ids: List of protein IDs
        model: ESM2 model
        tokenizer: ESM2 tokenizer
        device: torch device
        output_file: Path to save embeddings (.npz file)
        batch_size: Batch size for processing
        save_every: Save checkpoint every N sequences
    """
    embeddings_list = []
    ids_list = []
    
    print(f"\n Extracting embeddings with checkpointing...")
    
    for i in tqdm(range(0, len(sequences), batch_size)):
        batch_seqs = sequences[i:i + batch_size]
        batch_ids = protein_ids[i:i + batch_size]
        
        # Extract embeddings
        for seq, pid in zip(batch_seqs, batch_ids):
            emb = get_sequence_embedding(seq, model, tokenizer, device)
            embeddings_list.append(emb)
            ids_list.append(pid)
        
        # Save checkpoint
        if (i + batch_size) % save_every == 0:
            temp_file = f"{output_file}.checkpoint"
            np.savez_compressed(
                temp_file,
                embeddings=np.array(embeddings_list),
                protein_ids=np.array(ids_list)
            )
            print(f"\n✓ Checkpoint saved: {len(embeddings_list)} sequences")
    
    # Final save
    np.savez_compressed(
        output_file,
        embeddings=np.array(embeddings_list),
        protein_ids=np.array(ids_list)
    )
    
    print(f"\n All embeddings saved to {output_file}")
    print(f"   Total sequences: {len(embeddings_list)}")


# ============================================================================
# STEP 5: Load Pre-computed Embeddings
# ============================================================================

def load_embeddings(embeddings_file):
    """
    Load pre-computed embeddings from .npz file
    
    Args:
        embeddings_file: Path to .npz file
        
    Returns:
        embeddings: numpy array
        protein_ids: numpy array
    """
    data = np.load(embeddings_file)
    embeddings = data['embeddings']
    protein_ids = data['protein_ids']
    
    print(f"✓ Loaded embeddings: {embeddings.shape}")
    print(f"  Proteins: {len(protein_ids)}")
    
    return embeddings, protein_ids


# ============================================================================
# EXAMPLE USAGE
# ============================================================================

if __name__ == "__main__":
    # Example sequences
    example_sequences = [
        "MKTAYIAKQRQISFVKSHFSRQLEERLGLIEVQAPILSRVGDGTQDNLSGAEKAVQVKVK",
        "MRLLAPLSPLLALVAGAADDQLAQGLKDVTFLSVLEAGRTTLADPVELKRR",
        "MKKLLILTACFSILAAVGTQAGDTYAQYNQKLSDVVFPRDAFCAKDQCYVDCSQQFKD"
    ]
    
    print("=" * 70)
    print("ESM2 EMBEDDINGS EXTRACTION - DEMO")
    print("=" * 70)
    
    # Load model (use smallest for demo)
    model, tokenizer, device = load_esm2_model("facebook/esm2_t12_35M_UR50D")
    
    # Extract embeddings
    embeddings = extract_embeddings_batch(
        example_sequences,
        model,
        tokenizer,
        device,
        batch_size=2
    )
    
    print("\n" + "=" * 70)
    print(" EXTRACTION COMPLETE")
    print("=" * 70)
    print(f"Embeddings ready for classification!")
    print(f"Shape: {embeddings.shape}")
    print(f"Each protein is now represented by {embeddings.shape[1]} numbers")