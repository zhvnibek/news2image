import os
from telegram.ext import CommandHandler, MessageHandler, Filters
from gensim.models import KeyedVectors
from src.tg_bot import bot_config
from src.recommender import Recommender, Space
from src.config import Word2VecConfig, ImageConfig

# bot = bot_config.bot
updater = bot_config.updater
logger = bot_config.logger


def start(update, context):
    logger.info(update.effective_chat.id)
    context.bot.send_message(chat_id=update.effective_chat.id,
                             text="Hey, this is News2Image")


def test(update, context):
    original_text: str = update.message.text
    preds: list = recommender.predict(text=original_text)
    logger.info(preds)
    # context.bot.send_message(chat_id=update.effective_chat.id, text=test_msg)
    for pred in preds:
        try:
            image_id = pred[0]
            photo = open(os.path.join(images_folder, f'{image_id}.jpg'), mode='rb')
            context.bot.send_photo(chat_id=update.effective_chat.id, photo=photo)
        except Exception as e:
            logger.info(e)


if __name__ == '__main__':
    keyed_vectors = KeyedVectors.load_word2vec_format(fname=Word2VecConfig.get_word_vectors_filename(),
                                                      limit=Word2VecConfig.get_vocab_size(),
                                                      binary=True)
    space = Space(keyed_vectors)
    recommender = Recommender(space=space)
    recommender.set_image_subspaces(path=ImageConfig.get_image_subspaces_folder())
    images_folder = ImageConfig.get_images_folder()

    updater.start_polling()
    dispatcher = updater.dispatcher
    start_handler = CommandHandler('start', start)
    text_handler = MessageHandler(Filters.text, test)
    dispatcher.add_handler(start_handler)
    dispatcher.add_handler(text_handler)
