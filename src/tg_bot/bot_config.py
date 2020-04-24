import os
import logging
import telegram
from telegram import ParseMode

from telegram.ext import Updater

logging.basicConfig(format='%(levelname)s [%(asctime)s] [%(name)s] %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

parse_mode = ParseMode.MARKDOWN
TOKEN = os.environ.get('TOKEN')
updater = Updater(token=TOKEN, use_context=True)
bot = telegram.Bot(token=TOKEN)
