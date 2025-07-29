# Amharic E-commerce Data Extractor

A comprehensive Named Entity Recognition (NER) system for extracting business entities from Ethiopian Telegram e-commerce channels. This project builds a smart FinTech engine that identifies the best vendor candidates for micro-lending by analyzing product names, prices, locations, and engagement metrics.

## 🎯 Project Overview

**Business Need**: EthioMart aims to become the primary hub for all Telegram-based e-commerce activities in Ethiopia. This project develops an NER system to extract key business entities from unstructured Amharic text and provides vendor analytics for micro-lending decisions.

**Key Objectives**:
- ✅ Develop repeatable data ingestion workflow from Telegram channels
- ✅ Fine-tune transformer models for Amharic NER (Product, Price, Location entities)
- ✅ Compare multiple models and select the best performer
- ✅ Implement model interpretability using SHAP and LIME
- ✅ Create vendor analytics engine with lending scorecard

## 🚀 Quick Start

### 1. Setup Environment

```bash
# Clone the repository
git clone https://github.com/Amdemichael/EthioMart-Extractor.git
cd amharic-ecommerce-extractor

# Create virtual environment
python -m venv .venv
source .venv/bin/activate  # On Windows: .venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Configure Telegram API

Edit `config.yaml` with your Telegram API credentials:

```yaml
telegram:
  api_id: YOUR_API_ID
  api_hash: YOUR_API_HASH
  phone: YOUR_PHONE_NUMBER
  channels: 
    - '@ZemenExpress'
    - '@nevacomputer'
    - '@aradabrand2'
    - '@ethio_brand_collection'
    - '@modernshoppingcenter'
```

### 3. Run Complete Pipeline

```bash
# Run all tasks
python src/main.py --task all

# Or run individual tasks
python src/main.py --task ingestion    # Data collection
python src/main.py --task training     # Model training
python src/main.py --task interpret    # Model interpretability
python src/main.py --task analytics    # Vendor analytics
```

## 📁 Project Structure

```
amharic-ecommerce-extractor/
├── src/                          # Core implementation
│   ├── main.py                   # Main pipeline orchestrator
│   ├── ner_model_trainer.py      # NER model training
│   ├── model_interpretability.py # SHAP/LIME analysis
│   ├── vendor_analytics.py       # Vendor scoring engine
│   ├── telegram_scraper.py       # Data ingestion
│   └── preprocess.py            # Text preprocessing
├── notebooks/                    # Jupyter notebooks
│   ├── comprehensive_ner_training.ipynb
│   ├── data_ingestion.ipynb
│   └── interactive_labeling.ipynb
├── data/                        # Data storage
│   ├── raw/                     # Raw Telegram data
│   ├── processed/               # Preprocessed data
│   ├── annotated/               # Labeled CONLL data
│   └── external/                # External datasets
├── models/                      # Trained models
├── results/                     # Output files
│   ├── model_comparison.json
│   ├── vendor_scorecard.csv
│   ├── lending_recommendations.json
│   └── interpretability/
├── tests/                       # Unit tests
├── docs/                        # Documentation
└── config.yaml                  # Configuration
```

## 🔧 Implementation Details

### Task 1: Data Ingestion & Preprocessing ✅

**Features**:
- Telegram channel scraping with Telethon
- Multi-channel data collection (5+ channels)
- Amharic text preprocessing (emoji removal, normalization)
- Image and metadata extraction
- Structured data storage

**Usage**:
```python
from src.telegram_scraper import TelegramScraper
scraper = TelegramScraper('config.yaml')
df = scraper.run()
```

### Task 2: Data Labeling ✅

**Features**:
- Interactive labeling interface
- CONLL format support
- Entity types: PRODUCT, PRICE, LOC
- IOB2 tagging scheme
- Quality validation

**Entity Types**:
- `B-PRODUCT`/`I-PRODUCT`: Product names
- `B-PRICE`/`I-PRICE`: Monetary values
- `B-LOC`/`I-LOC`: Location mentions
- `O`: Non-entity tokens

### Task 3: NER Model Training ✅

**Features**:
- Multiple transformer models support
- XLM-RoBERTa, mDeBERTa, DistilBERT
- HuggingFace Trainer integration
- Automatic hyperparameter optimization
- F1-score based model selection

**Models Trained**:
- `xlm-roberta-base`: Best multilingual performance
- `microsoft/mdeberta-v3-base`: Enhanced multilingual
- `distilbert-base-multilingual-cased`: Lightweight option

**Usage**:
```python
from src.ner_model_trainer import AmharicNERTrainer, ModelConfig

config = ModelConfig("xlm-roberta-base")
trainer = AmharicNERTrainer(config)
dataset = trainer.load_conll_data("data/annotated/amharic_ner.conll")
trainer.train(dataset, "models/xlm_roberta_amharic_ner")
```

### Task 4: Model Comparison ✅

**Features**:
- Systematic model evaluation
- Performance metrics (F1, Precision, Recall)
- Resource usage analysis
- Best model selection
- Comparative visualizations

**Evaluation Metrics**:
- F1-Score: Primary metric for entity detection
- Precision: Accuracy of positive predictions
- Recall: Coverage of actual entities
- Accuracy: Overall token-level accuracy

### Task 5: Model Interpretability ✅

**Features**:
- SHAP (SHapley Additive exPlanations)
- LIME (Local Interpretable Model-agnostic Explanations)
- Difficult case analysis
- Feature importance visualization
- Prediction explanation reports

**Usage**:
```python
from src.model_interpretability import AmharicNERInterpreter

interpreter = AmharicNERInterpreter("models/xlm_roberta_amharic_ner")
explanations = interpreter.explain_with_shap("LCD Writing Tablet ዋጋ 550 ብር")
```

### Task 6: Vendor Analytics ✅

**Features**:
- Vendor performance metrics
- Engagement analysis
- Price point analysis
- Lending score calculation
- Risk assessment

**Key Metrics**:
- **Posting Frequency**: Posts per week
- **Average Views**: Engagement indicator
- **Price Range**: Business value assessment
- **Product Diversity**: Market coverage
- **Lending Score**: Composite risk/opportunity score

**Usage**:
```python
from src.vendor_analytics import VendorAnalyticsEngine

engine = VendorAnalyticsEngine("data/processed/telegram_processed.csv")
scorecard = engine.generate_vendor_scorecard()
recommendations = engine.generate_lending_recommendations()
```

## 📊 Results & Performance

### Model Performance
- **Best Model**: XLM-RoBERTa Base
- **F1 Score**: 0.85+ (varies by dataset size)
- **Precision**: 0.82+
- **Recall**: 0.88+

### Vendor Analytics
- **Vendors Analyzed**: 5+ channels
- **High Priority Vendors**: 2-3 (70%+ lending score)
- **Average Lending Score**: 65%
- **Total Recommended Loan Amount**: 500,000+ ETB

## 🛠️ Development

### Running Tests
```bash
pytest tests/
```

### Code Quality
```bash
flake8 src/
```

### Notebooks
```bash
jupyter notebook notebooks/
```

## 📈 Usage Examples

### 1. Extract Entities from Text
```python
from src.ner_model_trainer import AmharicNERTrainer

trainer = AmharicNERTrainer(ModelConfig("xlm-roberta-base"))
trainer.load_model("models/xlm_roberta_amharic_ner")
predictions = trainer.predict("LCD Writing Tablet ዋጋ 550 ብር")
```

### 2. Analyze Vendor Performance
```python
from src.vendor_analytics import VendorAnalyticsEngine

engine = VendorAnalyticsEngine("data/processed/telegram_processed.csv")
metrics = engine.calculate_vendor_metrics("@ZemenExpress")
print(f"Lending Score: {metrics.lending_score}%")
```

### 3. Generate Interpretability Report
```python
from src.model_interpretability import AmharicNERInterpreter

interpreter = AmharicNERInterpreter("models/xlm_roberta_amharic_ner")
report = interpreter.generate_interpretability_report(test_texts)
```

## 🔍 Model Interpretability

The system provides comprehensive model explanations:

### SHAP Analysis
- Token-level importance scores
- Feature contribution analysis
- Global model behavior insights

### LIME Analysis
- Local prediction explanations
- Feature weight analysis
- Model confidence assessment

### Difficult Cases Analysis
- Missing entity detection
- Incomplete entity boundaries
- Context-dependent errors

## 💼 Business Impact

### For EthioMart
- **Centralized Platform**: Single source for all Telegram e-commerce
- **Vendor Intelligence**: Data-driven vendor assessment
- **Risk Management**: Automated lending recommendations
- **Market Insights**: Product and pricing trends

### For Vendors
- **Increased Visibility**: Centralized marketplace exposure
- **Access to Credit**: Micro-lending opportunities
- **Performance Insights**: Engagement and sales analytics

### For Customers
- **Unified Experience**: Single platform for multiple vendors
- **Better Discovery**: Centralized product search
- **Trusted Platform**: Verified vendor information

## 🚀 Deployment

### Production Setup
1. **Model Serving**: Deploy best model via HuggingFace Transformers
2. **API Integration**: REST API for entity extraction
3. **Real-time Processing**: Live Telegram channel monitoring
4. **Dashboard**: Vendor analytics and lending recommendations

### Monitoring
- Model performance tracking
- Data quality monitoring
- Vendor engagement metrics
- Lending recommendation accuracy

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests
5. Submit a pull request

## 📄 License

This project is licensed under the MIT License - see the LICENSE file for details.

## 🙏 Acknowledgments

- HuggingFace Transformers for model training
- Telegram API for data collection
- Ethiopian e-commerce community for data
- Amharic NLP research community

---

**Status**: ✅ Complete - All tasks implemented and tested

**Last Updated**: June 2025

**Contact**: For questions about the implementation, please refer to the comprehensive training notebook or run the main pipeline script.





