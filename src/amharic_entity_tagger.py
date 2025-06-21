import pandas as pd
import re
import spacy
from spacy.training import offsets_to_biluo_tags
from pathlib import Path
import logging
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

class AmharicEntityTagger:
    def __init__(self):
        self.nlp = spacy.blank("xx")  # Multilingual model
        self.patterns = {
            'PRICE': [
                r'\b\d{1,3}(?:,\d{3})*\s*(?:ብር|ETB|birr)\b',
                r'Price[:፡]\s*\d{1,3}(?:,\d{3})*',
                r'ዋጋ\s*[:፡]\s*\d+'
            ],
            'LOC': [
                r'\b(Addis Ababa|Bole|ሐዋሳ|ሰሜን|አዲስ አበባ)\b',
                r'📍\s*(.+)'
            ],
            'PRODUCT': [
                r'📌\s*(.+)',
                r'\b(iPhone\s?\d*|Galaxy S\d+|S23|ስልክ|Humidifier|Baby)\b'
            ],
            'CONTACT': [
                r'09\d{8}',
                r'@[\w]+',
                r'Contact\s*[:፡]\s*(.+)'
            ]
        }

    def extract_entities(self, text):
        """Improved with entity text capture"""
        entities = []
        for label, patterns in self.patterns.items():
            for pat in patterns:
                for match in re.finditer(pat, str(text), re.IGNORECASE):
                    entities.append((match.start(), match.end(), label))
        return sorted(entities, key=lambda x: x[0])

    def process_to_conll(self, input_csv, output_path, sample_size=None):
        """Main processing function"""
        try:
            df = pd.read_csv(input_csv)
            texts = df['preprocessed_text'].dropna().tolist()
            
            if sample_size:
                texts = texts[:sample_size]
                
            with open(output_path, 'w', encoding='utf-8') as f:
                for text in tqdm(texts, desc="Processing texts"):
                    entities = self.extract_entities(text)
                    doc = self.nlp.make_doc(text)
                    tags = offsets_to_biluo_tags(doc, entities)
                    
                    for token, tag in zip(doc, tags):
                        f.write(f"{token.text}\t{tag if tag != '-' else 'O'}\n")
                    f.write("\n")
            
            logging.info(f"Successfully processed {len(texts)} texts to {output_path}")
            return True
            
        except Exception as e:
            logging.error(f"Error processing file: {str(e)}")
            return False

def main():
    tagger = AmharicEntityTagger()
    tagger.process_to_conll(
        input_csv="../data/processed/telegram_data_final.csv",
        output_path="../data/annotated/ner_data.conll",
        sample_size=500  # Remove for full dataset
    )

if __name__ == "__main__":
    main()