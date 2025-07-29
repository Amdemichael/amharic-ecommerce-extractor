import os
import asyncio
from telethon import TelegramClient, events
from telethon.tl.types import MessageMediaPhoto
import pandas as pd
import yaml
from datetime import datetime
import logging
from typing import List, Dict, Optional

class TelegramScraper:
    """Scrape Ethiopian e-commerce Telegram channels"""
    
    def __init__(self, config_path: str = 'config.yaml'):
        self.config = self._load_config(config_path)
        self._ensure_dirs_exist()
        self.client = None
        
    def _load_config(self, config_path: str) -> Dict:
        """Load YAML configuration"""
        try:
            with open(config_path, 'r') as f:
                config = yaml.safe_load(f)
                
            required_keys = ['api_id', 'api_hash', 'channels']
            if not all(k in config['telegram'] for k in required_keys):
                raise ValueError("Missing required configuration keys")
                
            return config
        except Exception as e:
            logging.error(f"Config loading failed: {str(e)}")
            raise

    def _ensure_dirs_exist(self) -> None:
        """Create required directories"""
        os.makedirs('data/raw', exist_ok=True)
        os.makedirs('data/images', exist_ok=True)
        os.makedirs('data/processed', exist_ok=True)

    async def _process_message(self, message) -> Optional[Dict]:
        """Process a single Telegram message"""
        try:
            msg_data = {
                'message_id': message.id,
                'channel': message.chat.username or str(message.chat.id),
                'text': message.text or '',
                'timestamp': message.date.isoformat(),
                'views': message.views or 0,
                'sender': message.sender_id if message.sender_id else None,
                'image_path': '',
                'has_product': 0
            }

            if message.media and isinstance(message.media, MessageMediaPhoto):
                img_filename = f"{msg_data['channel']}_{message.id}.jpg"
                img_path = os.path.join('data/images', img_filename)
                await message.download_media(file=img_path)
                msg_data['image_path'] = img_path

            product_keywords = ['ዋጋ', 'ብር', 'ETB', 'ሽያጭ', 'አስገባ']
            if any(kw in msg_data['text'] for kw in product_keywords):
                msg_data['has_product'] = 1

            return msg_data
        except Exception as e:
            logging.error(f"Error processing message {message.id}: {str(e)}")
            return None

    async def scrape_channel(self, channel: str, limit: int = 200) -> List[Dict]:
        """Scrape messages from a single channel"""
        messages = []
        try:
            async for message in self.client.iter_messages(channel, limit=limit):
                processed = await self._process_message(message)
                if processed:
                    messages.append(processed)
                    
                if len(messages) % 20 == 0:
                    logging.info(f"Collected {len(messages)} messages from {channel}")
                    
            return messages
        except Exception as e:
            logging.error(f"Error scraping {channel}: {str(e)}")
            return []

    async def run(self) -> pd.DataFrame:
        """Main execution method"""
        self.client = TelegramClient(
            'ethiomart_session',
            self.config['telegram']['api_id'],
            self.config['telegram']['api_hash']
        )
        
        await self.client.start()
        
        all_messages = []
        for channel in self.config['telegram']['channels']:
            logging.info(f"Starting scrape for {channel}")
            messages = await self.scrape_channel(channel)
            all_messages.extend(messages)
            logging.info(f"Finished {channel}. Total messages: {len(messages)}")
            
        df = pd.DataFrame(all_messages)
        raw_path = os.path.join('data/raw', f'telegram_raw_{datetime.now().strftime("%Y%m%d")}.csv')
        df.to_csv(raw_path, index=False, encoding='utf-8')
        
        await self.client.disconnect()
        return df
