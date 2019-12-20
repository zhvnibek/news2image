import sys
import os
from gensim.models import KeyedVectors
from image.encoder import ImageEncoder
# from text.encoder import TextEncoder
from image.captioning.captioner import Captioner
from text.summarizer import CaptionSummarizer
from common.space import Space
from config import w2v_file, vocab_limit
from config import checkpoint_path, word_map_path, device
from config import google_open_images
from config import get_logger
from config import stop_words

sys.path.append(os.path.join(sys.path[0],'image','captioning')) # add models.py

logger = get_logger('recommender')

class Recommender:

    def __init__(self):
        pass
