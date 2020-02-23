import os
from telegram.ext import CommandHandler, MessageHandler, Filters
from src.tg_bot import bot_config
from src.tg_bot.resolver import Resolver, Response

# bot = bot_config.bot
updater = bot_config.updater
logger = bot_config.logger


def start(update, context):
    logger.info(update.effective_chat.id)
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text="Hey, this is News2Image")


def process(update, context):
    logger.info(f'Processing text from {update.effective_chat.id}: {update.message.text[:20]}')
    options = None
    response: Response = resolver.resolve(text=update.message.text, options=options)
    context.bot.send_message(chat_id=update.effective_chat.id, text=response.get('summary'))
    context.bot.send_message(chat_id=update.effective_chat.id, text=response.get('keywords'))
    collage_loc = response.get('collage_loc')
    photo = open(file=collage_loc, mode='rb')
    context.bot.send_photo(chat_id=update.effective_chat.id,  photo=photo, caption='Images')
    if os.path.exists(collage_loc):
        os.remove(collage_loc)

if __name__ == '__main__':
    resolver = Resolver()
    updater.start_polling()
    dispatcher = updater.dispatcher
    start_handler = CommandHandler('start', start)
    text_handler = MessageHandler(Filters.text, process)
    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(text_handler)
