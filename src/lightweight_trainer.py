#!/usr/bin/env python3
"""
Lightweight NER Trainer for Limited Resources
"""

import os
import sys
import logging
import gc
import torch
from typing import Dict, List, Tuple, Optional
from dataclasses import dataclass

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split
import evaluate

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class LightweightConfig:
    """Lightweight configuration for limited resources"""
    model_name: str = "distilbert-base-multilingual-cased"  # Smaller model
    learning_rate: float = 5e-5
    num_epochs: int = 1  # Reduced epochs
    batch_size: int = 4   # Very small batch size
    max_length: int = 256 # Reduced sequence length
    eval_strategy: str = "no"  # No evaluation during training
    save_strategy: str = "no"  # No saving during training
    load_best_model_at_end: bool = False
    greater_is_better: bool = True

class LightweightNERTrainer:
    """Lightweight trainer for limited resources"""
    
    def __init__(self, config: LightweightConfig):
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
        
        # Force CPU to save memory
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        torch.set_num_threads(1)  # Limit CPU threads
        
    def load_conll_data(self, file_path: str, max_samples: int = 50) -> DatasetDict:
        """Load CONLL data with limited samples"""
        logger.info(f"Loading CONLL data from {file_path} (max {max_samples} samples)")
        
        sentences = []
        current_sentence = []
        
        with open(file_path, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.strip()
                if line == "":
                    if current_sentence:
                        sentences.append(current_sentence)
                        current_sentence = []
                        if len(sentences) >= max_samples:
                            break
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
        
        # Use only a small portion for training
        if len(data['tokens']) > max_samples:
            data['tokens'] = data['tokens'][:max_samples]
            data['ner_tags'] = data['ner_tags'][:max_samples]
            data['id'] = data['id'][:max_samples]
        
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
        
        logger.info(f"Loaded {len(train_dataset)} training, {len(val_dataset)} validation samples")
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def tokenize_and_align_labels(self, examples):
        """Tokenize and align labels with minimal memory usage"""
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
    
    def train(self, dataset: DatasetDict, output_dir: str):
        """Train with minimal resource usage"""
        logger.info(f"Training {self.config.model_name} with lightweight config")
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
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
        
        # Setup training arguments for minimal resources
        training_args = TrainingArguments(
            output_dir=output_dir,
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,
            load_best_model_at_end=self.config.load_best_model_at_end,
            greater_is_better=self.config.greater_is_better,
            push_to_hub=False,
            logging_steps=10,
            save_total_limit=1,
            dataloader_pin_memory=False,  # Save memory
            dataloader_num_workers=0,     # No multiprocessing
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
        )
        
        # Train the model
        logger.info("Starting lightweight training...")
        train_result = self.trainer.train()
        
        # Save model and tokenizer
        self.trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        # Clear memory
        gc.collect()
        torch.cuda.empty_cache() if torch.cuda.is_available() else None
        
        logger.info(f"Lightweight training completed. Loss: {train_result.training_loss:.4f}")
        return train_result
    
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

def main():
    """Main function for lightweight training"""
    print("🚀 Starting Lightweight NER Training")
    print("This version uses minimal resources for limited systems")
    
    # Configure for minimal resources
    config = LightweightConfig(
        model_name="distilbert-base-multilingual-cased",  # Smaller model
        batch_size=2,  # Very small batch
        num_epochs=1,  # Single epoch
        max_length=128  # Short sequences
    )
    
    # Initialize trainer
    trainer = LightweightNERTrainer(config)
    
    # Load data
    conll_path = 'data/annotated/amharic_ner.conll'
    if os.path.exists(conll_path):
        dataset = trainer.load_conll_data(conll_path, max_samples=30)
        
        # Train model
        output_dir = "models/lightweight_ner"
        os.makedirs(output_dir, exist_ok=True)
        
        train_result = trainer.train(dataset, output_dir)
        
        # Test predictions
        test_text = "LCD Writing Tablet ዋጋ 550 ብር"
        predictions = trainer.predict(test_text)
        
        print("\n✅ Lightweight training completed!")
        print(f"Training loss: {train_result.training_loss:.4f}")
        print(f"Test predictions: {predictions}")
        
    else:
        print("❌ CONLL data not found")

if __name__ == "__main__":
    main() 