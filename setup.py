from setuptools import setup, find_packages

setup(
    name="CAFA6-Protein-Function-Prediction",
    version="0.1.0",
    author="05Menna",
    description="CAFA 6 Protein Function Prediction",
    url="https://github.com/05Menna/CAFA6-Protein-Function-Prediction",
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
