"""
Automatic Project Setup Script
================================
This script creates the complete CAFA 6 project structure automatically

Usage:
    python auto_setup.py

Author: Menna
Date: 2025
"""

import os
import subprocess
from pathlib import Path

# ============================================================================
# Configuration
# ============================================================================

PROJECT_NAME = "CAFA6-Protein-Function-Prediction"
GITHUB_USERNAME = "menna890"  # ⚠️ CHANGE THIS TO YOUR USERNAME!

# ============================================================================
# Colors for terminal output
# ============================================================================

class Colors:
    HEADER = '\033[95m'
    BLUE = '\033[94m'
    GREEN = '\033[92m'
    YELLOW = '\033[93m'
    RED = '\033[91m'
    END = '\033[0m'
    BOLD = '\033[1m'

def print_header(text):
    print(f"\n{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{text:^70}{Colors.END}")
    print(f"{Colors.HEADER}{Colors.BOLD}{'='*70}{Colors.END}\n")

def print_success(text):
    print(f"{Colors.GREEN}✓ {text}{Colors.END}")

def print_warning(text):
    print(f"{Colors.YELLOW}⚠ {text}{Colors.END}")

def print_error(text):
    print(f"{Colors.RED}✗ {text}{Colors.END}")

def print_info(text):
    print(f"{Colors.BLUE}ℹ {text}{Colors.END}")

# ============================================================================
# Step 1: Create Directory Structure
# ============================================================================

def create_directories():
    """Create all project directories"""
    print_header("STEP 1: Creating Directory Structure")
    
    directories = [
        "data/raw/Train",
        "data/raw/Test",
        "data/processed",
        "data/embeddings",
        "src/data",
        "src/features",
        "src/models",
        "src/evaluation",
        "src/utils",
        "notebooks",
        "scripts",
        "models/baseline",
        "models/advanced",
        "configs",
        "tests",
        "outputs/predictions",
        "outputs/figures",
        "outputs/reports",
        "docs"
    ]
    
    for directory in directories:
        Path(directory).mkdir(parents=True, exist_ok=True)
        print_success(f"Created: {directory}")
    
    print_success("All directories created!")

# ============================================================================
# Step 2: Create Python Package Files
# ============================================================================

def create_init_files():
    """Create __init__.py files"""
    print_header("STEP 2: Creating Package Files")
    
    init_files = [
        "src/__init__.py",
        "src/data/__init__.py",
        "src/features/__init__.py",
        "src/models/__init__.py",
        "src/evaluation/__init__.py",
        "src/utils/__init__.py",
        "tests/__init__.py",
    ]
    
    for init_file in init_files:
        with open(init_file, 'w') as f:
            f.write('"""Package initialization"""\n')
        print_success(f"Created: {init_file}")

# ============================================================================
# Step 3: Create Configuration Files
# ============================================================================

def create_requirements():
    """Create requirements.txt"""
    requirements = """# Core Libraries
numpy>=1.21.0
pandas>=1.3.0
scipy>=1.7.0

# Machine Learning
scikit-learn>=1.0.0
xgboost>=1.7.0
lightgbm>=3.3.0

# Deep Learning
torch>=2.0.0
transformers>=4.30.0

# Bioinformatics
biopython>=1.79

# Utilities
tqdm>=4.65.0
pyyaml>=6.0
matplotlib>=3.5.0
seaborn>=0.12.0

# Development
jupyter>=1.0.0
pytest>=7.0.0
black>=22.0.0
"""
    
    with open("requirements.txt", "w") as f:
        f.write(requirements)
    print_success("Created: requirements.txt")


def create_gitignore():
    """Create .gitignore"""
    gitignore = """# Data files - DON'T UPLOAD!
data/raw/
data/processed/
data/embeddings/
*.fasta
*.tsv
*.csv

# Models - Too large
models/*.pkl
models/*.h5
*.pkl

# Outputs
outputs/predictions/
outputs/figures/

# Python
__pycache__/
*.py[cod]
*.so
.Python
build/
dist/
*.egg-info/

# Virtual Environments
.env
.venv
venv/
ENV/

# Jupyter
.ipynb_checkpoints

# IDEs
.vscode/
.idea/
*.swp
.DS_Store

# Logs
*.log
logs/
"""
    
    with open(".gitignore", "w") as f:
        f.write(gitignore)
    print_success("Created: .gitignore")


def create_readme():
    """Create README.md"""
    readme = f"""# 🧬 CAFA 6 Protein Function Prediction

Predict protein function from amino acid sequences using ESM2 protein language models and XGBoost classifiers.

## 🎯 Competition

[CAFA 6 Protein Function Prediction](https://www.kaggle.com/competitions/cafa-6-protein-function-prediction) on Kaggle

## 📊 Approach

- **Embeddings:** ESM2 protein language model
- **Classification:** Multi-label XGBoost  
- **Evaluation:** CAFA-specific metrics (F1, Precision, Recall)

## 🚀 Quick Start

### Installation

```bash
# Clone repository
git clone https://github.com/{GITHUB_USERNAME}/{PROJECT_NAME}.git
cd {PROJECT_NAME}

# Install dependencies
pip install -r requirements.txt
```

### Download Data

```bash
# Using Kaggle API
kaggle competitions download -c cafa-6-protein-function-prediction
unzip cafa-6-protein-function-prediction.zip -d data/raw/
```

### Run Pipeline

```bash
# Extract embeddings
python scripts/extract_embeddings.py

# Train model
python scripts/train.py

# Make predictions
python scripts/predict.py
```

## 📁 Project Structure

```
├── data/               # Data files (not tracked)
├── src/                # Source code
│   ├── data/          # Data loading
│   ├── features/      # Feature extraction
│   ├── models/        # Model definitions
│   └── utils/         # Utilities
├── notebooks/          # Jupyter notebooks
├── scripts/           # Executable scripts
├── models/            # Trained models (not tracked)
└── outputs/           # Results and reports
```

## 🔬 Methodology

1. **Data Loading**: Parse FASTA and TSV files
2. **Embeddings**: Extract ESM2 embeddings (480-dim vectors)
3. **Classification**: Multi-label Binary Relevance with XGBoost
4. **Evaluation**: CAFA metrics with cross-validation

## 📈 Results

| Model | F1-Macro | F1-Micro | Precision | Recall |
|-------|----------|----------|-----------|--------|
| Baseline | TBD | TBD | TBD | TBD |

## 🛠️ Technologies

- Python 3.9+
- PyTorch
- Transformers (HuggingFace)
- XGBoost
- scikit-learn
- BioPython

## 👤 Author

Menna - [@{GITHUB_USERNAME}](https://github.com/{GITHUB_USERNAME})

## 📝 License

MIT License

## 🙏 Acknowledgments

- CAFA competition organizers
- Meta AI for ESM2
- Kaggle community
"""
    
    with open("README.md", "w", encoding="utf-8") as f:
        f.write(readme)
    print_success("Created: README.md")


def create_config_yaml():
    """Create config.yaml"""
    config = """# CAFA 6 Configuration

data:
  raw_dir: "data/raw"
  processed_dir: "data/processed"
  embeddings_dir: "data/embeddings"

model:
  esm2_name: "facebook/esm2_t12_35M_UR50D"
  top_n_go_terms: 1000
  batch_size: 8

training:
  validation_split: 0.2
  n_estimators: 100
  random_state: 42

output:
  models_dir: "models"
  predictions_dir: "outputs/predictions"
"""
    
    with open("configs/config.yaml", "w") as f:
        f.write(config)
    print_success("Created: configs/config.yaml")


def create_setup_py():
    """Create setup.py"""
    setup = f'''from setuptools import setup, find_packages

setup(
    name="cafa6-protein-prediction",
    version="0.1.0",
    author="Menna",
    description="CAFA 6 Protein Function Prediction",
    url="https://github.com/{GITHUB_USERNAME}/{PROJECT_NAME}",
    packages=find_packages(),
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.21.0",
        "pandas>=1.3.0",
        "scikit-learn>=1.0.0",
        "xgboost>=1.7.0",
        "torch>=2.0.0",
        "transformers>=4.30.0",
        "biopython>=1.79",
    ],
)
'''
    
    with open("setup.py", "w") as f:
        f.write(setup)
    print_success("Created: setup.py")

# ============================================================================
# Step 4: Create Placeholder Files
# ============================================================================

def create_placeholder_scripts():
    """Create placeholder Python scripts"""
    print_header("STEP 3: Creating Placeholder Scripts")
    
    # Data loader placeholder
    data_loader = '''"""Data loading utilities for CAFA 6"""

def load_protein_sequences(fasta_file):
    """Load sequences from FASTA file"""
    # TODO: Implement data loading
    pass

def load_go_annotations(tsv_file):
    """Load GO annotations from TSV"""
    # TODO: Implement annotation loading
    pass
'''
    
    with open("src/data/data_loader.py", "w") as f:
        f.write(data_loader)
    print_success("Created: src/data/data_loader.py")
    
    # Embeddings placeholder
    embeddings = '''"""ESM2 embeddings extraction"""

def load_esm2_model(model_name="facebook/esm2_t12_35M_UR50D"):
    """Load ESM2 model from HuggingFace"""
    # TODO: Implement model loading
    pass

def extract_embeddings(sequences, model):
    """Extract embeddings for sequences"""
    # TODO: Implement embedding extraction
    pass
'''
    
    with open("src/features/esm2_embeddings.py", "w") as f:
        f.write(embeddings)
    print_success("Created: src/features/esm2_embeddings.py")
    
    # Training script placeholder
    train_script = '''"""Training pipeline for CAFA 6"""

import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent.parent))

def main():
    print("Training pipeline - TODO: Implement")
    pass

if __name__ == "__main__":
    main()
'''
    
    with open("scripts/train.py", "w") as f:
        f.write(train_script)
    print_success("Created: scripts/train.py")

# ============================================================================
# Step 5: Git Initialization
# ============================================================================

def initialize_git():
    """Initialize Git repository"""
    print_header("STEP 4: Initializing Git Repository")
    
    try:
        # Check if already a git repo
        if Path(".git").exists():
            print_warning("Git repository already initialized")
            return
        
        # Initialize git
        subprocess.run(["git", "init"], check=True)
        print_success("Git initialized")
        
        # Add all files
        subprocess.run(["git", "add", "."], check=True)
        print_success("Files staged")
        
        # Initial commit
        subprocess.run([
            "git", "commit", "-m", 
            "Initial project structure with placeholders"
        ], check=True)
        print_success("Initial commit created")
        
        # Instructions for GitHub
        print_info(f"\nTo push to GitHub, run:")
        print(f"  git remote add origin https://github.com/{GITHUB_USERNAME}/{PROJECT_NAME}.git")
        print(f"  git branch -M main")
        print(f"  git push -u origin main")
        
    except subprocess.CalledProcessError as e:
        print_error(f"Git error: {e}")
    except FileNotFoundError:
        print_error("Git not found. Please install Git first!")

# ============================================================================
# Main Function
# ============================================================================

def main():
    """Main setup function"""
    print_header("CAFA 6 PROJECT AUTO SETUP")
    print_info(f"Project: {PROJECT_NAME}")
    print_info(f"GitHub: {GITHUB_USERNAME}")
    print()
    
    # Create structure
    create_directories()
    create_init_files()
    
    print_header("STEP 3: Creating Configuration Files")
    create_requirements()
    create_gitignore()
    create_readme()
    create_config_yaml()
    create_setup_py()
    create_placeholder_scripts()
    
    # Git initialization
    initialize_git()
    
    # Final instructions
    print_header("✅ SETUP COMPLETE!")
    print_success("Project structure created successfully!")
    print()
    print_info("Next steps:")
    print("  1. Create repository on GitHub")
    print("  2. Run: git remote add origin <YOUR_REPO_URL>")
    print("  3. Run: git push -u origin main")
    print("  4. Install dependencies: pip install -r requirements.txt")
    print("  5. Download CAFA 6 data from Kaggle")
    print("  6. Start coding in src/ directory!")
    print()
    print(f"{Colors.BOLD}Happy coding! 🚀{Colors.END}\n")

if __name__ == "__main__":
    main()