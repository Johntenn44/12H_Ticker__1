import os
import asyncio
from datetime import datetime
import pytz
from telegram import Bot

async def main():
    TELEGRAM_TOKEN = os.getenv('TELEGRAM_BOT_TOKEN')
    TELEGRAM_CHAT_ID = os.getenv('TELEGRAM_CHAT_ID')

    if not TELEGRAM_TOKEN or not TELEGRAM_CHAT_ID:
        print("Missing TELEGRAM_TOKEN or TELEGRAM_CHAT_ID environment variables")
        return

    timezone = pytz.timezone('Africa/Lagos')
    now = datetime.now(timezone)
    current_hour = now.hour
    print(f"Current hour: {current_hour}")

    if 0 <= current_hour < 5:
        message = f"It's between 12AM and 5AM, Good morning! ☀️ Have a great day!"
    elif 18 <= current_hour < 24:
        message = "It's between 6PM and 12AM, Good night! 🌙 Sleep well!"
    else:
        print("No message to send at this hour.")
        return

    print(f"Sending message: {message}")
    bot = Bot(token=TELEGRAM_TOKEN)
    await bot.send_message(chat_id=TELEGRAM_CHAT_ID, text=message)

if __name__ == "__main__":
    asyncio.run(main())