import sys
import os
from gensim.models import KeyedVectors
from text.encoder import TextEncoder
from text.summarizer import NewsSummarizer
from common.space import Space
from config import w2v_file, vocab_limit
from config import get_logger

sys.path.append(os.path.join(sys.path[0],'image','captioning')) # add models.py

logger = get_logger('recommender')

class Recommender:

    def __init__(self, space: Space):
        self.space = space
        
