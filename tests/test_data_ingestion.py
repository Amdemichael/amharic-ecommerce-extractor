import pytest
from unittest.mock import AsyncMock, patch
from src.data_ingestion import fetch_messages

@pytest.mark.asyncio
async def test_fetch_messages():
    config = {
        'telegram': {'api_id': 'mock_id', 'api_hash': 'mock_hash'},
        'channels': ['@TestChannel']
    }
    with patch('telethon.TelegramClient.start', new=AsyncMock()):
        with patch('telethon.TelegramClient.iter_messages', new=AsyncMock()) as mock_iter:
            mock_iter.return_value = [
                type('Message', (), {
                    'id': 1, 'text': 'Test', 'date': '2025-06-19',
                    'views': 100, 'sender_id': 123, 'photo': None
                })
            ]
            messages = await fetch_messages(config)
            assert len(messages) == 1
            assert messages[0]['message_id'] == 1