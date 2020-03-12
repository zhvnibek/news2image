import os
from telegram.ext import CommandHandler, MessageHandler, Filters, CallbackQueryHandler
from telegram import InlineKeyboardButton, InlineKeyboardMarkup
from src.tg_bot import bot_config
from src.tg_bot.resolver import Resolver, Response

# bot = bot_config.bot
updater = bot_config.updater
logger = bot_config.logger


def start(update, context):
    # logger.info(update.effective_chat.id)
    # context.bot.send_message(chat_id=update.effective_chat.id, text="Hey, this is News2Image")
    keyboard = [[InlineKeyboardButton("Option 1", callback_data='1'),
                 InlineKeyboardButton("Option 2", callback_data='2')],

                [InlineKeyboardButton("Option 3", callback_data='3')]]

    reply_markup = InlineKeyboardMarkup(keyboard)
    msg = 'Hey, this is News2Image. You can send here any news text to get some images which describe it.'
    update.message.reply_text(msg, reply_markup=reply_markup)


def process(update, context):
    logger.info(f'Processing text from {update.effective_chat.id}: {update.message.text[:20]}')
    params = None
    chat_id = update.effective_chat.id
    response: Response = resolver.resolve(text=update.message.text, params=params)
    context.bot.send_message(chat_id=chat_id, text=response.get('summary'))
    context.bot.send_message(chat_id=chat_id, text=response.get('keywords'))
    context.bot.send_message(chat_id=chat_id, text=response.get('captions'))
    collage_loc = response.get('collage_loc')
    try:
        photo = open(file=collage_loc, mode='rb')
        context.bot.send_photo(chat_id=chat_id,  photo=photo, caption='Images')
    except Exception as e:
        logger.error(e)
    if os.path.exists(collage_loc):
        os.remove(collage_loc)
    img_captions = "Here should be the captions.."
    keyboard = [[InlineKeyboardButton("Show Captions", callback_data=img_captions)]]
    reply_markup = InlineKeyboardMarkup(keyboard)
    msg = 'More:'
    context.bot.send_message(chat_id=chat_id, text=msg, reply_markup=reply_markup)


def button(update, context):
    query = update.callback_query

    query.edit_message_text(text="Captions: {}".format(query.data))


if __name__ == '__main__':
    resolver = Resolver()
    updater.start_polling()
    dispatcher = updater.dispatcher
    start_handler = CommandHandler('start', start)
    text_handler = MessageHandler(Filters.text, process)
    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(text_handler)
    dispatcher.add_handler(CallbackQueryHandler(button))
