import os
import json
import logging
import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
from seqeval.metrics import classification_report, accuracy_score
import torch
from torch.utils.data import DataLoader
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class ModelConfig:
    """Configuration for NER model training"""
    model_name: str
    learning_rate: float = 5e-5
    num_epochs: int = 3
    batch_size: int = 16
    max_length: int = 512
    warmup_steps: int = 500
    weight_decay: float = 0.01
    eval_strategy: str = "epoch"
    save_strategy: str = "epoch"
    load_best_model_at_end: bool = True
    greater_is_better: bool = True

class AmharicNERTrainer:
    """Trainer for Amharic NER models"""
    
    def __init__(self, config: ModelConfig):
        self.config = config
        self.tokenizer = None
        self.model = None
        self.trainer = None
        self.label2id = {
            'O': 0,
            'B-PRODUCT': 1, 'I-PRODUCT': 2,
            'B-PRICE': 3, 'I-PRICE': 4,
            'B-LOC': 5, 'I-LOC': 6
        }
        self.id2label = {v: k for k, v in self.label2id.items()}
        
    def load_conll_data(self, file_path: str) -> DatasetDict:
        """Load CONLL format data and convert to HuggingFace dataset"""
        logger.info(f"Loading CONLL data from {file_path}")
        
        sentences = []
        current_sentence = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                else:
                    parts = line.split('\t')
                    if len(parts) >= 2:
                        token, label = parts[0], parts[1]
                        current_sentence.append((token, label))
            
            if current_sentence:
                sentences.append(current_sentence)
        
        # Convert to dataset format
        data = {
            'tokens': [],
            'ner_tags': [],
            'id': []
        }
        
        for i, sentence in enumerate(sentences):
            tokens = [token for token, _ in sentence]
            labels = [self.label2id.get(label, 0) for _, label in sentence]
            
            data['tokens'].append(tokens)
            data['ner_tags'].append(labels)
            data['id'].append(i)
        
        # Split into train/validation
        train_data, val_data = train_test_split(
            list(zip(data['tokens'], data['ner_tags'], data['id'])), 
            test_size=0.2, 
            random_state=42
        )
        
        train_dataset = Dataset.from_dict({
            'tokens': [item[0] for item in train_data],
            'ner_tags': [item[1] for item in train_data],
            'id': [item[2] for item in train_data]
        })
        
        val_dataset = Dataset.from_dict({
            'tokens': [item[0] for item in val_data],
            'ner_tags': [item[1] for item in val_data],
            'id': [item[2] for item in val_data]
        })
        
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def tokenize_and_align_labels(self, examples):
        """Tokenize and align labels with tokenizer output"""
        tokenized_inputs = self.tokenizer(
            examples["tokens"],
            truncation=True,
            is_split_into_words=True,
            max_length=self.config.max_length,
            padding="max_length"
        )
        
        labels = []
        for i, label in enumerate(examples["ner_tags"]):
            word_ids = tokenized_inputs.word_ids(batch_index=i)
            previous_word_idx = None
            label_ids = []
            
            for word_idx in word_ids:
                if word_idx is None:
                    label_ids.append(-100)
                elif word_idx != previous_word_idx:
                    label_ids.append(label[word_idx])
                else:
                    label_ids.append(-100)
                previous_word_idx = word_idx
            
            labels.append(label_ids)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def compute_metrics(self, eval_preds):
        """Compute evaluation metrics"""
        predictions, labels = eval_preds
        predictions = np.argmax(predictions, axis=2)
        
        true_predictions = [
            [self.id2label[p] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        true_labels = [
            [self.id2label[l] for (p, l) in zip(prediction, label) if l != -100]
            for prediction, label in zip(predictions, labels)
        ]
        
        results = evaluate.load("seqeval").compute(
            predictions=true_predictions, 
            references=true_labels
        )
        
        return {
            "precision": results["overall_precision"],
            "recall": results["overall_recall"],
            "f1": results["overall_f1"],
            "accuracy": results["overall_accuracy"],
        }
    
    def train(self, dataset: DatasetDict, output_dir: str):
        """Train the NER model"""
        logger.info(f"Training {self.config.model_name} for Amharic NER")
        
        # Load tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(self.config.model_name)
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.config.model_name,
            num_labels=len(self.label2id),
            id2label=self.id2label,
            label2id=self.label2id
        )
        
        # Tokenize datasets
        tokenized_datasets = dataset.map(
            self.tokenize_and_align_labels,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        # Setup training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            weight_decay=self.config.weight_decay,
            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            greater_is_better=self.config.greater_is_better,
            push_to_hub=False,
            logging_dir=f"{output_dir}/logs",
            logging_steps=100,
            save_total_limit=3,
        )
        
        # Setup data collator
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        # Initialize trainer
        self.trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=tokenized_datasets["train"],
            eval_dataset=tokenized_datasets["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics,
        )
        
        # Train the model
        logger.info("Starting training...")
        train_result = self.trainer.train()
        
        # Save model and tokenizer
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Log training results
        logger.info(f"Training completed. Results: {train_result}")
        
        return train_result
    
    def evaluate(self, test_dataset: Optional[Dataset] = None):
        """Evaluate the trained model"""
        if test_dataset is None:
            test_dataset = self.trainer.eval_dataset
        
        results = self.trainer.evaluate(test_dataset)
        logger.info(f"Evaluation results: {results}")
        return results
    
    def predict(self, text: str) -> List[Tuple[str, str]]:
        """Predict entities in a given text"""
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not trained. Call train() first.")
        
        # Tokenize input
        tokens = text.split()
        inputs = self.tokenizer(
            tokens,
            is_split_into_words=True,
            return_tensors="pt",
            truncation=True,
            max_length=self.config.max_length
        )
        
        # Get predictions
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

def train_multiple_models():
    """Train multiple models for comparison"""
    models = [
        ModelConfig("xlm-roberta-base", learning_rate=5e-5, num_epochs=3),
        ModelConfig("microsoft/mdeberta-v3-base", learning_rate=3e-5, num_epochs=3),
        ModelConfig("distilbert-base-multilingual-cased", learning_rate=5e-5, num_epochs=3),
    ]
    
    results = {}
    
    for model_config in models:
        logger.info(f"Training {model_config.model_name}")
        
        trainer = AmharicNERTrainer(model_config)
        
        # Load data
        dataset = trainer.load_conll_data("data/annotated/amharic_ner.conll")
        
        # Train model
        output_dir = f"models/{model_config.model_name.replace('/', '_')}"
        os.makedirs(output_dir, exist_ok=True)
        
        train_result = trainer.train(dataset, output_dir)
        
        # Evaluate
        eval_result = trainer.evaluate()
        
        results[model_config.model_name] = {
            'train_loss': train_result.training_loss,
            'eval_f1': eval_result['eval_f1'],
            'eval_precision': eval_result['eval_precision'],
            'eval_recall': eval_result['eval_recall'],
            'eval_accuracy': eval_result['eval_accuracy']
        }
        
        logger.info(f"Results for {model_config.model_name}: {results[model_config.model_name]}")
    
    # Save comparison results
    with open("results/model_comparison.json", "w") as f:
        json.dump(results, f, indent=2)
    
    return results

if __name__ == "__main__":
    # Train a single model for testing
    config = ModelConfig("xlm-roberta-base")
    trainer = AmharicNERTrainer(config)
    
    dataset = trainer.load_conll_data("data/annotated/amharic_ner.conll")
    trainer.train(dataset, "models/xlm_roberta_test") 