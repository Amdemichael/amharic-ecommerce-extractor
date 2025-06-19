import yaml
import os
import pandas as pd
from telethon.sync import TelegramClient
import asyncio
import logging

logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)


def load_config():
    with open('config.yaml', 'r') as f:
        return yaml.safe_load(f)


async def fetch_messages(config):
    api_id = config['telegram']['api_id']
    api_hash = config['telegram']['api_hash']
    channels = config['channels']
    os.makedirs('data/images/telegram_images', exist_ok=True)
    client = TelegramClient('session_name', api_id, api_hash)
    await client.start()
    data = []

    for channel in channels:
        logging.info(f"Scraping {channel}")
        try:
            async for message in client.iter_messages(channel, limit=100):
                msg_data = {
                    'message_id': message.id,
                    'channel': channel,
                    'text': message.text if message.text else '',
                    'timestamp': message.date,
                    'views': message.views if message.views else 0,
                    'sender': message.sender_id if message.sender_id else 'unknown',
                    'image_path': ''
                }
                if message.photo:
                    image_path = (
                        f"data/images/telegram_images/"
                        f"{channel[1:]}_{message.id}.jpg"
                    )
                    await message.download_media(file=image_path)
                    msg_data['image_path'] = image_path
                data.append(msg_data)
        except Exception as e:
            logging.error(f"Error scraping {channel}: {e}")
    await client.disconnect()
    return data


if __name__ == "__main__":
    config = load_config()
    messages = asyncio.run(fetch_messages(config))
    df = pd.DataFrame(messages)
    df.to_csv('data/raw/telegram_data.csv', index=False, encoding='utf-8')
    logging.info(f"Collected {len(df)} messages")
