# Amharic E-commerce Data Extractor

This project builds an Amharic NER system to extract entities (Product, Price, Location) from Ethiopian Telegram e-commerce channels for EthioMart. It includes data ingestion, preprocessing, model fine-tuning, comparison, interpretability, and vendor analytics.

## Setup
1. Clone the repository:
   ```bash
   git clone https://github.com/Amdemichael/EthioMart-Extractor.git

2. Install dependencies
```python
    pip install -r requirements.txt
```

3. Add Telegram API credentials to config.yaml

## CI/CD
- GitHub Actions workflow (.github/workflows/ci-cd.yml) runs:
    - Linting with flake8

    - Unit tests with pytest

    - Data validation and artifact archiving

    - Model deployment (on tagged releases)

## Structure
- .github/workflows/: CI/CD workflows

- src/: Python scripts for tasks

- tests/: Unit tests

- data/: Datasets

- models/: Fine-tuned models

- notebooks/: Exploratory notebooks

- docs/: Reports

- results/: Metrics and visualizations

## Create Directories
```bash
mkdir -p .github/workflows src tests data/raw data/processed data/labeled data/external data/images/telegram_images
mkdir -p models/xlm_roberta models/distilbert models/mbert models/checkpoints
mkdir -p notebooks docs results/shap_plots results/lime_reports
```







