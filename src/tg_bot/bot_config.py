import os
import logging
import telegram
from telegram.ext import Updater

logging.basicConfig(format='%(levelname)s [%(asctime)s] [%(name)s] %(message)s',
                    datefmt='%d/%m/%Y %H:%M:%S',
                    level=logging.INFO)
logger = logging.getLogger(__name__)

TOKEN = os.environ.get('TOKEN', '1076785258:AAHwK0Bea7jOhtzea4Qpvd3eOUSLloMoc_E')
updater = Updater(token=TOKEN, use_context=True)
bot = telegram.Bot(token=TOKEN)