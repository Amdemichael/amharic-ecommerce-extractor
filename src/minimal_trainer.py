#!/usr/bin/env python3
"""
Minimal NER Trainer - No Plotting, No Disk Saving
"""

import os
import sys
import gc
import torch
from typing import List, Tuple
from dataclasses import dataclass

# Force CPU and minimal resources
os.environ['CUDA_VISIBLE_DEVICES'] = ''
torch.set_num_threads(1)

from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    TrainingArguments,
    Trainer,
    DataCollatorForTokenClassification
)
from datasets import Dataset, DatasetDict
from sklearn.model_selection import train_test_split

@dataclass
class MinimalConfig:
    """Minimal configuration - no disk saving, no plotting"""
    model_name: str = "distilbert-base-multilingual-cased"
    learning_rate: float = 5e-5
    num_epochs: int = 1
    batch_size: int = 1
    max_length: int = 32  # Very short sequences
    eval_strategy: str = "no"
    save_strategy: str = "no"

class MinimalNERTrainer:
    """Minimal trainer - no disk operations, no plotting"""
    
    def __init__(self, config: MinimalConfig):
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
        
    def load_conll_data(self, file_path: str, max_samples: int = 10) -> DatasetDict:
        """Load CONLL data with very limited samples"""
        print(f"Loading CONLL data from {file_path} (max {max_samples} samples)")
        
        sentences = []
        current_sentence = []
        
        try:
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
        except Exception as e:
            print(f"Error reading file: {e}")
            # Create dummy data if file doesn't exist
            sentences = [
                [("LCD", "B-PRODUCT"), ("Writing", "I-PRODUCT"), ("Tablet", "I-PRODUCT"), ("ዋጋ", "B-PRICE"), ("550", "I-PRICE"), ("ብር", "I-PRICE")],
                [("ስልክ", "B-PRODUCT"), ("ዋጋ", "B-PRICE"), ("2500", "I-PRICE"), ("ETB", "I-PRICE")]
            ]
        
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
        if len(data['tokens']) > 1:
            train_data, val_data = train_test_split(
                list(zip(data['tokens'], data['ner_tags'], data['id'])), 
                test_size=0.2, 
                random_state=42
            )
        else:
            # If only one sample, duplicate it
            train_data = [(data['tokens'][0], data['ner_tags'][0], data['id'][0])]
            val_data = [(data['tokens'][0], data['ner_tags'][0], data['id'][0])]
        
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
        
        print(f"Loaded {len(train_dataset)} training, {len(val_dataset)} validation samples")
        return DatasetDict({
            'train': train_dataset,
            'validation': val_dataset
        })
    
    def tokenize_and_align_labels(self, examples):
        """Tokenize and align labels"""
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
    
    def train(self, dataset: DatasetDict):
        """Train without saving to disk"""
        print(f"Training {self.config.model_name} with minimal config")
        
        # Clear memory
        gc.collect()
        
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
        
        # Setup training arguments - NO SAVING
        training_args = TrainingArguments(
            output_dir="/tmp/minimal_ner",  # Temporary directory
            learning_rate=self.config.learning_rate,
            per_device_train_batch_size=self.config.batch_size,
            per_device_eval_batch_size=self.config.batch_size,
            num_train_epochs=self.config.num_epochs,
            eval_strategy=self.config.eval_strategy,
            save_strategy=self.config.save_strategy,  # NO SAVING
            load_best_model_at_end=False,
            greater_is_better=True,
            push_to_hub=False,
            logging_steps=1,
            save_total_limit=0,  # Don't save any checkpoints
            dataloader_pin_memory=False,
            dataloader_num_workers=0,
            report_to=None,  # No reporting
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
        print("Starting minimal training...")
        train_result = self.trainer.train()
        
        # Clear memory
        gc.collect()
        
        print(f"Minimal training completed. Loss: {train_result.training_loss:.4f}")
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
    """Main function for minimal training"""
    print("🚀 Starting Minimal NER Training")
    print("This version uses minimal resources and NO DISK SAVING")
    
    # Configure for minimal resources
    config = MinimalConfig(
        model_name="distilbert-base-multilingual-cased",
        batch_size=1,  # Minimum batch size
        num_epochs=1,  # Single epoch
        max_length=32  # Very short sequences
    )
    
    # Initialize trainer
    trainer = MinimalNERTrainer(config)
    
    # Load data
    conll_path = '../data/annotated/amharic_ner.conll'
    dataset = trainer.load_conll_data(conll_path, max_samples=10)
    
    # Train model (no saving)
    train_result = trainer.train(dataset)
    
    # Test predictions
    test_text = "LCD Writing Tablet ዋጋ 550 ብር"
    predictions = trainer.predict(test_text)
    
    print("\n✅ Minimal training completed!")
    print(f"Training loss: {train_result.training_loss:.4f}")
    print(f"Test predictions: {predictions}")
    print("\nNote: Model was trained in memory only (no disk saving)")

if __name__ == "__main__":
    main() 