import os
import json
import logging
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Dict, Tuple, Optional
import torch
from transformers import AutoTokenizer, AutoModelForTokenClassification
import shap
import lime
from lime.lime_text import LimeTextExplainer
from sklearn.metrics import confusion_matrix
import warnings
warnings.filterwarnings('ignore')

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AmharicNERInterpreter:
    """Model interpretability for Amharic NER models"""
    
    def __init__(self, model_path: str):
        self.model_path = model_path
        self.tokenizer = None
        self.model = None
        self.label2id = {
            'O': 0,
            'B-PRODUCT': 1, 'I-PRODUCT': 2,
            'B-PRICE': 3, 'I-PRICE': 4,
            'B-LOC': 5, 'I-LOC': 6
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        self._load_model()
        
    def _load_model(self):
        """Load the trained model and tokenizer"""
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_path)
            self.model = AutoModelForTokenClassification.from_pretrained(self.model_path)
            self.model.eval()
            logger.info(f"Model loaded from {self.model_path}")
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            raise
    
    def predict_proba(self, text: str) -> np.ndarray:
        """Get prediction probabilities for SHAP"""
        tokens = text.split()
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            probs = torch.softmax(outputs.logits, dim=-1)
        
        return probs.numpy()
    
    def predict(self, text: str) -> List[Tuple[str, str]]:
        """Predict entities in text"""
        tokens = text.split()
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=512
        )
        
        with torch.no_grad():
            outputs = self.model(**inputs)
            predictions = torch.argmax(outputs.logits, dim=2)
        
        # Align predictions with tokens
        word_ids = inputs.word_ids(0)
        aligned_predictions = []
        
        for i, word_idx in enumerate(word_ids):
            if word_idx is not None:
                label = self.id2label[predictions[0][i].item()]
                aligned_predictions.append((tokens[word_idx], label))
        
        return aligned_predictions
    
    def explain_with_shap(self, text: str, save_path: str = None) -> Dict:
        """Explain model predictions using SHAP"""
        logger.info("Generating SHAP explanations...")
        
        # Create explainer
        explainer = shap.Explainer(self.predict_proba, self.tokenizer)
        
        # Generate explanations
        shap_values = explainer([text])
        
        # Process results
        tokens = text.split()
        explanations = {}
        
        for i, token in enumerate(tokens):
            if i < len(shap_values.values[0]):
                token_importance = shap_values.values[0][i]
                explanations[token] = {
                    'importance': token_importance.tolist(),
                    'base_value': shap_values.base_values[0][i].tolist()
                }
        
        # Save results
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(explanations, f, indent=2, ensure_ascii=False)
        
        return explanations
    
    def explain_with_lime(self, text: str, num_features: int = 10, save_path: str = None) -> Dict:
        """Explain model predictions using LIME"""
        logger.info("Generating LIME explanations...")
        
        # Create explainer
        explainer = LimeTextExplainer(class_names=list(self.label2id.keys()))
        
        # Define prediction function for LIME
        def predict_fn(texts):
            results = []
            for text in texts:
                tokens = text.split()
                inputs = self.tokenizer(
                    tokens,
                    is_split_into_words=True,
                    return_tensors="pt",
                    truncation=True,
                    max_length=512
                )
                
                with torch.no_grad():
                    outputs = self.model(**inputs)
                    probs = torch.softmax(outputs.logits, dim=-1)
                
                # Get max probability for each token
                max_probs = torch.max(probs, dim=-1)[0]
                results.append(max_probs.mean().item())
            
            return np.array(results)
        
        # Generate explanation
        exp = explainer.explain_instance(
            text, 
            predict_fn, 
            num_features=num_features,
            labels=[0, 1, 2, 3, 4, 5, 6]  # All label indices
        )
        
        # Process results
        lime_explanation = {
            'text': text,
            'feature_weights': {},
            'local_pred': exp.local_pred,
            'score': exp.score
        }
        
        for label_idx in range(len(self.label2id)):
            if label_idx in exp.local_exp:
                feature_weights = exp.local_exp[label_idx]
                lime_explanation['feature_weights'][self.id2label[label_idx]] = [
                    (feature, weight) for feature, weight in feature_weights
                ]
        
        # Save results
        if save_path:
            os.makedirs(os.path.dirname(save_path), exist_ok=True)
            with open(save_path, 'w', encoding='utf-8') as f:
                json.dump(lime_explanation, f, indent=2, ensure_ascii=False)
        
        return lime_explanation
    
    def analyze_difficult_cases(self, test_texts: List[str], save_dir: str = "results/interpretability"):
        """Analyze cases where the model struggles"""
        logger.info("Analyzing difficult cases...")
        
        os.makedirs(save_dir, exist_ok=True)
        difficult_cases = []
        
        for i, text in enumerate(test_texts):
            predictions = self.predict(text)
            
            # Identify potential issues
            issues = []
            
            # Check for O predictions in product/price contexts
            product_keywords = ['ዋጋ', 'ብር', 'ETB', 'ሽያጭ']
            price_keywords = ['ብር', 'ETB', 'birr']
            
            has_product_context = any(kw in text for kw in product_keywords)
            has_price_context = any(kw in text for kw in price_keywords)
            
            predicted_products = [token for token, label in predictions if 'PRODUCT' in label]
            predicted_prices = [token for token, label in predictions if 'PRICE' in label]
            
            if has_product_context and not predicted_products:
                issues.append("Missing product detection")
            
            if has_price_context and not predicted_prices:
                issues.append("Missing price detection")
            
            # Check for inconsistent labeling
            labels = [label for _, label in predictions]
            if 'B-PRODUCT' in labels and 'I-PRODUCT' not in labels:
                issues.append("Incomplete product entity")
            
            if 'B-PRICE' in labels and 'I-PRICE' not in labels:
                issues.append("Incomplete price entity")
            
            if issues:
                difficult_cases.append({
                    'text': text,
                    'predictions': predictions,
                    'issues': issues,
                    'case_id': i
                })
        
        # Save difficult cases
        with open(f"{save_dir}/difficult_cases.json", 'w', encoding='utf-8') as f:
            json.dump(difficult_cases, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Found {len(difficult_cases)} difficult cases")
        return difficult_cases
    
    def generate_interpretability_report(self, test_texts: List[str], save_dir: str = "results/interpretability"):
        """Generate comprehensive interpretability report"""
        logger.info("Generating interpretability report...")
        
        os.makedirs(save_dir, exist_ok=True)
        
        # Analyze difficult cases
        difficult_cases = self.analyze_difficult_cases(test_texts, save_dir)
        
        # Generate SHAP explanations for sample texts
        shap_explanations = {}
        for i, text in enumerate(test_texts[:5]):  # Limit to 5 for performance
            try:
                shap_exp = self.explain_with_shap(
                    text, 
                    f"{save_dir}/shap_explanation_{i}.json"
                )
                shap_explanations[f"text_{i}"] = shap_exp
            except Exception as e:
                logger.warning(f"SHAP explanation failed for text {i}: {e}")
        
        # Generate LIME explanations
        lime_explanations = {}
        for i, text in enumerate(test_texts[:5]):
            try:
                lime_exp = self.explain_with_lime(
                    text, 
                    save_path=f"{save_dir}/lime_explanation_{i}.json"
                )
                lime_explanations[f"text_{i}"] = lime_exp
            except Exception as e:
                logger.warning(f"LIME explanation failed for text {i}: {e}")
        
        # Create summary report
        report = {
            'model_path': self.model_path,
            'total_texts_analyzed': len(test_texts),
            'difficult_cases_count': len(difficult_cases),
            'shap_explanations_count': len(shap_explanations),
            'lime_explanations_count': len(lime_explanations),
            'difficult_cases_summary': {
                'missing_product_detection': sum(1 for case in difficult_cases if 'Missing product detection' in case['issues']),
                'missing_price_detection': sum(1 for case in difficult_cases if 'Missing price detection' in case['issues']),
                'incomplete_entities': sum(1 for case in difficult_cases if any('Incomplete' in issue for issue in case['issues']))
            }
        }
        
        # Save report
        with open(f"{save_dir}/interpretability_report.json", 'w', encoding='utf-8') as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        # Generate visualizations
        self._create_visualizations(difficult_cases, save_dir)
        
        logger.info(f"Interpretability report saved to {save_dir}")
        return report
    
    def _create_visualizations(self, difficult_cases: List[Dict], save_dir: str):
        """Create visualizations for interpretability analysis"""
        
        # Issue frequency plot
        all_issues = []
        for case in difficult_cases:
            all_issues.extend(case['issues'])
        
        issue_counts = pd.Series(all_issues).value_counts()
        
        plt.figure(figsize=(10, 6))
        issue_counts.plot(kind='bar')
        plt.title('Frequency of Model Issues')
        plt.xlabel('Issue Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{save_dir}/issue_frequency.png", dpi=300, bbox_inches='tight')
        plt.close()
        
        # Entity distribution in difficult cases
        entity_counts = {'PRODUCT': 0, 'PRICE': 0, 'LOC': 0}
        for case in difficult_cases:
            for _, label in case['predictions']:
                if 'PRODUCT' in label:
                    entity_counts['PRODUCT'] += 1
                elif 'PRICE' in label:
                    entity_counts['PRICE'] += 1
                elif 'LOC' in label:
                    entity_counts['LOC'] += 1
        
        plt.figure(figsize=(8, 6))
        plt.pie(entity_counts.values(), labels=entity_counts.keys(), autopct='%1.1f%%')
        plt.title('Entity Distribution in Difficult Cases')
        plt.savefig(f"{save_dir}/entity_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()

def main():
    """Main function to run interpretability analysis"""
    # Example usage
    model_path = "models/xlm_roberta_base"  # Update with actual model path
    
    # Sample test texts
    test_texts = [
        "LCD Writing Tablet ዋጋ 550 ብር አድራሻ መገናኛ_መሰረት_ደፋር_ሞል",
        "ስልክ ዋጋ 2500 ETB ቦሌ አድራሻ",
        "ኮምፒዩተር ላፕቶፕ ዋጋ 15000 ብር",
        "ቴሌቪዥን ዋጋ 8000 ETB ፒያሳ",
        "ስልክ ዋጋ አድራሻ"
    ]
    
    try:
        interpreter = AmharicNERInterpreter(model_path)
        report = interpreter.generate_interpretability_report(test_texts)
        print("Interpretability analysis completed successfully!")
        print(f"Report summary: {report}")
    except Exception as e:
        logger.error(f"Error in interpretability analysis: {e}")

if __name__ == "__main__":
    main() 