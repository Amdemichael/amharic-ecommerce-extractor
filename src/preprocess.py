import pandas as pd
import argparse
import re
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def preprocess_amharic(text):
    if not isinstance(text, str):
        return ''
    text = re.sub(r'[^\w\s]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    text = re.sub(r'(\d+)\s*ብር', r'\1 ETB', text)
    return text


def validate_csv(input_path):
    df = pd.read_csv(input_path)
    required_columns = [
        'message_id', 'channel', 'text', 'preprocessed_text',
        'timestamp', 'views', 'sender', 'image_path'
    ]
    if not all(col in df.columns for col in required_columns):
        raise ValueError("Missing required columns")
    if not df['message_id'].notnull().all():
        raise ValueError("Missing message IDs")
    logging.info("CSV validation passed")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Preprocess Telegram data")
    parser.add_argument(
        '--validate',
        action='store_true',
        help='Validate processed CSV'
    )
    parser.add_argument(
        '--input',
        default='../data/raw/telegram_data.csv',
        help='Input CSV path'
    )
    parser.add_argument(
        '--output',
        default='../data/processed/telegram_data_final.csv',
        help='Output CSV path'
    )
    args = parser.parse_args()

    if args.validate:
        validate_csv(args.output)
    else:
        df = pd.read_csv(args.input)
        df['preprocessed_text'] = df['text'].apply(preprocess_amharic)
        df.to_csv(args.output, index=False, encoding='utf-8')
        logging.info(
            f"Saved {len(df)} preprocessed messages to {args.output}"
        )
