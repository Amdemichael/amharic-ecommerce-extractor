import pandas as pd
import argparse
import re
import logging
import os
from emoji import demojize  # For emoji handling

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)

def preprocess_amharic(text):
    if not isinstance(text, str):
        return ''
    
    # Step 1: Remove all emojis and symbols
    text = demojize(text)
    text = re.sub(r':[a-z_]+:', '', text)  # Remove emoji codes
    text = re.sub(r'[\U0001F600-\U0001F6FF\U0001F300-\U0001F5FF]', '', text)  # Additional emoji removal
    
    # Step 2: Clean Telegram-specific patterns
    telegram_patterns = [
        r'[📌📍💥👍⚡️⚠️🏢🔖💬]',  # Icons
        r'\.{3,}',  # Ellipsis (3+ dots)
        r'�+',      # Replacement chars
        r'\*{2,}',  # Multiple asterisks
        r'_{2,}',   # Multiple underscores
        r'#\w+'     # Hashtags
    ]
    for pattern in telegram_patterns:
        text = re.sub(pattern, '', text)
    
    # Step 3: Standardize text
    text = re.sub(r'[^\w\s]', ' ', text)  # Keep words/whitespace
    text = re.sub(r'(\d+)\s*(ብር|ETB|birr|Br)', r'\1 ETB', text, flags=re.IGNORECASE)
    text = re.sub(r'\s+', ' ', text).strip()
    
    return text

def validate_csv(input_path):
    df = pd.read_csv(input_path)
    required_columns = ['message_id', 'channel', 'message', 'preprocessed_text', 
                       'timestamp', 'views', 'sender', 'image_path']
    if not all(col in df.columns for col in required_columns):
        raise ValueError("❌ Missing required columns")
    if df['message_id'].isnull().any():
        raise ValueError("❌ NULL values in message_id")
    logging.info("✅ CSV validation passed")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Telegram text")
    parser.add_argument('--validate', action='store_true', help='Validate processed CSV')
    parser.add_argument('--input', default='../data/raw/telegram_data.csv', help='Input CSV path')
    parser.add_argument('--output', default='../data/processed/telegram_data_final.csv', help='Output CSV path')
    
    args = parser.parse_args()

    if args.validate:
        validate_csv(args.output)
    else:
        df = pd.read_csv(args.input)
        
        # Handle NULL messages and preprocess
        df['preprocessed_text'] = df['message'].fillna('').apply(preprocess_amharic)
        
        # Ensure output directory exists
        os.makedirs(os.path.dirname(args.output), exist_ok=True)
        df.to_csv(args.output, index=False, encoding='utf-8')
        
        logging.info(f"✅ Saved {len(df)} preprocessed messages to {args.output}")