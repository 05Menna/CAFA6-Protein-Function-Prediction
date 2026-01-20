# 🧬 CAFA 6 Protein Function Prediction

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
git clone https://github.com/menna890/CAFA6-Protein-Function-Prediction.git
cd CAFA6-Protein-Function-Prediction

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

Menna - [@menna890](https://github.com/menna890)

## 📝 License

MIT License

## 🙏 Acknowledgments

- CAFA competition organizers
- Meta AI for ESM2
- Kaggle community
