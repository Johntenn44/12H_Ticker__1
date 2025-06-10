import os
import asyncio
from datetime import datetime
import pytz
from telegram import Bot
import logging

# Configure logging
logging.basicConfig(level=logging.INFO)

async def main():
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        logging.error("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID environment variables")
        return

    timezone = pytz.timezone('Africa/Lagos')
    now = datetime.now(timezone)
    current_hour = now.hour
    logging.info(f"Current hour: {current_hour}")

    if 0 <= current_hour < 5:
        message = (
            "It's between 12AM and 5AM, Good morning! ☀️ Have a great day!\n"
            "---Confirmed trends, only on second mention---\n"
            "If CP touches low purple after trend identification, trend facing reversal may occur.\n"
            "If CP touches trend purple before trend identification, reversal may occur.\n"
            "Green before purple, continues."
        )
    elif 18 <= current_hour < 24:
        message = "It's between 6PM and 12AM, Good night! 🌙 Sleep well!"
    else:
        logging.info("No message to send at this hour.")
        return

    logging.info(f"Sending message: {message}")
    bot = Bot(token=TELEGRAM_TOKEN)
    try:
        await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)
        logging.info("Message sent successfully.")
    except Exception as e:
        logging.error(f"Failed to send message: {e}")

if __name__ == "__main__":
    asyncio.run(main())
