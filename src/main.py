#!/usr/bin/env python3
"""
Amharic E-commerce Data Extractor - Main Pipeline

This script orchestrates all tasks for the Amharic NER project:
1. Data ingestion and preprocessing
2. NER model training and comparison
3. Model interpretability analysis
4. Vendor analytics and lending scorecard

Usage:
    python src/main.py --task all
    python src/main.py --task train
    python src/main.py --task interpret
    python src/main.py --task analytics
"""

import os
import sys
import argparse
import logging
import json
from datetime import datetime
from typing import Dict, List

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from ner_model_trainer import AmharicNERTrainer, ModelConfig, train_multiple_models
from model_interpretability import AmharicNERInterpreter
from vendor_analytics import VendorAnalyticsEngine
from telegram_scraper import TelegramScraper
from preprocess import preprocess_amharic

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('logs/main_pipeline.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class AmharicNERPipeline:
    """Main pipeline for Amharic NER project"""
    
    def __init__(self):
        self.results = {}
        self._ensure_directories()
    
    def _ensure_directories(self):
        """Create necessary directories"""
        directories = [
            'data/raw', 'data/processed', 'data/annotated',
            'models', 'results', 'logs',
            'results/interpretability', 'results/shap_plots', 'results/lime_reports'
        ]
        
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
    
    def run_data_ingestion(self) -> Dict:
        """Task 1: Data ingestion and preprocessing"""
        logger.info("=== Task 1: Data Ingestion and Preprocessing ===")
        
        try:
            # Initialize scraper
            scraper = TelegramScraper('config.yaml')
            
            # Run scraping
            logger.info("Starting Telegram data scraping...")
            df = scraper.run()
            
            # Preprocess data
            logger.info("Preprocessing data...")
            df['preprocessed_text'] = df['text'].fillna('').apply(preprocess_amharic)
            
            # Save processed data
            output_path = 'data/processed/telegram_processed.csv'
            df.to_csv(output_path, index=False, encoding='utf-8')
            
            result = {
                'status': 'success',
                'total_messages': len(df),
                'channels': df['channel'].nunique(),
                'output_path': output_path
            }
            
            logger.info(f"Data ingestion completed: {len(df)} messages from {df['channel'].nunique()} channels")
            return result
            
        except Exception as e:
            logger.error(f"Error in data ingestion: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def run_model_training(self) -> Dict:
        """Task 3: NER model training"""
        logger.info("=== Task 3: NER Model Training ===")
        
        try:
            # Check if CONLL data exists
            conll_path = 'data/annotated/amharic_ner.conll'
            if not os.path.exists(conll_path):
                logger.warning("CONLL data not found. Using sample data...")
                # Create sample CONLL data if not available
                self._create_sample_conll_data(conll_path)
            
            # Train multiple models
            logger.info("Training multiple models for comparison...")
            results = train_multiple_models()
            
            # Find best model
            best_model = max(results.keys(), key=lambda k: results[k]['eval_f1'])
            best_f1 = results[best_model]['eval_f1']
            
            result = {
                'status': 'success',
                'models_trained': len(results),
                'best_model': best_model,
                'best_f1_score': best_f1,
                'all_results': results
            }
            
            logger.info(f"Model training completed. Best model: {best_model} (F1: {best_f1:.4f})")
            return result
            
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def _create_sample_conll_data(self, output_path: str):
        """Create sample CONLL data for testing"""
        sample_data = [
            "LCD\tB-PRODUCT",
            "Writing\tI-PRODUCT", 
            "Tablet\tI-PRODUCT",
            "ዋጋ\tO",
            "550\tB-PRICE",
            "ብር\tI-PRICE",
            "አድራሻ\tO",
            "መገናኛ_መሰረት_ደፋር_ሞል\tB-LOC",
            "",
            "ስልክ\tB-PRODUCT",
            "ዋጋ\tO",
            "2500\tB-PRICE",
            "ETB\tI-PRICE",
            "ቦሌ\tB-LOC",
            "አድራሻ\tO"
        ]
        
        with open(output_path, 'w', encoding='utf-8') as f:
            f.write('\n'.join(sample_data))
        
        logger.info(f"Created sample CONLL data: {output_path}")
    
    def run_interpretability_analysis(self) -> Dict:
        """Task 5: Model interpretability"""
        logger.info("=== Task 5: Model Interpretability ===")
        
        try:
            # Find best model
            model_comparison_path = 'results/model_comparison.json'
            if os.path.exists(model_comparison_path):
                with open(model_comparison_path, 'r') as f:
                    results = json.load(f)
                best_model = max(results.keys(), key=lambda k: results[k]['eval_f1'])
            else:
                best_model = 'xlm-roberta-base'
            
            model_path = f"models/{best_model.replace('/', '_')}"
            
            if not os.path.exists(model_path):
                logger.warning(f"Model {model_path} not found. Skipping interpretability...")
                return {'status': 'skipped', 'reason': 'Model not found'}
            
            # Initialize interpreter
            interpreter = AmharicNERInterpreter(model_path)
            
            # Test texts
            test_texts = [
                "LCD Writing Tablet ዋጋ 550 ብር አድራሻ መገናኛ_መሰረት_ደፋር_ሞል",
                "ስልክ ዋጋ 2500 ETB ቦሌ አድራሻ",
                "ኮምፒዩተር ላፕቶፕ ዋጋ 15000 ብር"
            ]
            
            # Generate interpretability report
            report = interpreter.generate_interpretability_report(test_texts)
            
            result = {
                'status': 'success',
                'model_used': best_model,
                'texts_analyzed': report['total_texts_analyzed'],
                'difficult_cases': report['difficult_cases_count'],
                'shap_explanations': report['shap_explanations_count'],
                'lime_explanations': report['lime_explanations_count']
            }
            
            logger.info("Interpretability analysis completed successfully")
            return result
            
        except Exception as e:
            logger.error(f"Error in interpretability analysis: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def run_vendor_analytics(self) -> Dict:
        """Task 6: Vendor analytics and lending scorecard"""
        logger.info("=== Task 6: Vendor Analytics ===")
        
        try:
            # Check if processed data exists
            data_path = 'data/processed/telegram_processed.csv'
            if not os.path.exists(data_path):
                logger.warning("Processed data not found. Skipping vendor analytics...")
                return {'status': 'skipped', 'reason': 'Data not found'}
            
            # Initialize analytics engine
            engine = VendorAnalyticsEngine(data_path)
            
            # Generate scorecard
            scorecard = engine.generate_vendor_scorecard()
            
            # Create visualizations
            engine.create_visualizations()
            
            # Generate lending recommendations
            recommendations = engine.generate_lending_recommendations()
            
            result = {
                'status': 'success',
                'vendors_analyzed': len(engine.vendor_metrics),
                'high_priority_vendors': recommendations['summary']['high_priority_count'],
                'medium_priority_vendors': recommendations['summary']['medium_priority_count'],
                'low_priority_vendors': recommendations['summary']['low_priority_count'],
                'avg_lending_score': recommendations['summary']['avg_lending_score'],
                'total_recommended_loan_amount': recommendations['summary']['total_recommended_loan_amount']
            }
            
            logger.info(f"Vendor analytics completed: {len(engine.vendor_metrics)} vendors analyzed")
            return result
            
        except Exception as e:
            logger.error(f"Error in vendor analytics: {e}")
            return {'status': 'error', 'error': str(e)}
    
    def run_all_tasks(self) -> Dict:
        """Run all tasks in sequence"""
        logger.info("=== Running Complete Amharic NER Pipeline ===")
        
        results = {
            'timestamp': datetime.now().isoformat(),
            'tasks': {}
        }
        
        # Task 1: Data ingestion
        results['tasks']['data_ingestion'] = self.run_data_ingestion()
        
        # Task 3: Model training
        results['tasks']['model_training'] = self.run_model_training()
        
        # Task 5: Interpretability
        results['tasks']['interpretability'] = self.run_interpretability_analysis()
        
        # Task 6: Vendor analytics
        results['tasks']['vendor_analytics'] = self.run_vendor_analytics()
        
        # Save overall results
        with open('results/pipeline_results.json', 'w') as f:
            json.dump(results, f, indent=2)
        
        # Print summary
        self._print_summary(results)
        
        return results
    
    def _print_summary(self, results: Dict):
        """Print pipeline execution summary"""
        print("\n" + "="*60)
        print("AMHARIC NER PIPELINE - EXECUTION SUMMARY")
        print("="*60)
        
        for task_name, task_result in results['tasks'].items():
            status = task_result.get('status', 'unknown')
            print(f"\n{task_name.upper().replace('_', ' ')}: {status.upper()}")
            
            if status == 'success':
                if task_name == 'data_ingestion':
                    print(f"  Messages processed: {task_result.get('total_messages', 0)}")
                    print(f"  Channels: {task_result.get('channels', 0)}")
                elif task_name == 'model_training':
                    print(f"  Models trained: {task_result.get('models_trained', 0)}")
                    print(f"  Best model: {task_result.get('best_model', 'N/A')}")
                    print(f"  Best F1 score: {task_result.get('best_f1_score', 0):.4f}")
                elif task_name == 'interpretability':
                    print(f"  Texts analyzed: {task_result.get('texts_analyzed', 0)}")
                    print(f"  Difficult cases: {task_result.get('difficult_cases', 0)}")
                elif task_name == 'vendor_analytics':
                    print(f"  Vendors analyzed: {task_result.get('vendors_analyzed', 0)}")
                    print(f"  High priority: {task_result.get('high_priority_vendors', 0)}")
                    print(f"  Avg lending score: {task_result.get('avg_lending_score', 0):.2f}%")
            elif status == 'error':
                print(f"  Error: {task_result.get('error', 'Unknown error')}")
            elif status == 'skipped':
                print(f"  Reason: {task_result.get('reason', 'Unknown reason')}")
        
        print("\n" + "="*60)
        print("Pipeline execution completed!")
        print("Check 'results/' directory for detailed outputs.")
        print("="*60)

def main():
    """Main function"""
    parser = argparse.ArgumentParser(description='Amharic NER Pipeline')
    parser.add_argument('--task', choices=['all', 'ingestion', 'training', 'interpret', 'analytics'], 
                       default='all', help='Task to run')
    
    args = parser.parse_args()
    
    pipeline = AmharicNERPipeline()
    
    if args.task == 'all':
        pipeline.run_all_tasks()
    elif args.task == 'ingestion':
        pipeline.run_data_ingestion()
    elif args.task == 'training':
        pipeline.run_model_training()
    elif args.task == 'interpret':
        pipeline.run_interpretability_analysis()
    elif args.task == 'analytics':
        pipeline.run_vendor_analytics()

if __name__ == "__main__":
    main() 