import sys
import os
import numpy as np
from collections import Counter
from gensim.models import KeyedVectors
from image.encoder import ImageEncoder
from text.encoder import TextEncoder
from image.captioning.captioner import Captioner
from text.summarizer import CaptionSummarizer, NewsSummarizer
from common.space import Space
from config import w2v_file, vocab_limit
from config import checkpoint_path, word_map_path, device
from config import google_open_images_folder
from config import get_logger
from config import stop_words

sys.path.append(os.path.join(sys.path[0],'image','captioning')) # add models.py

logger = get_logger('generate_subspaces')

""" Generate word subspaces from text """
def generate_subspaces_from_text():
    pass

""" Generate word subspaces from images """
def generate_subspaces_from_images(images_folder: str):
    for i in os.listdir(images_folder):
        image_filename = os.path.join(images_folder, i)
        word_subspace = image_encoder.create_subspace(image_filename)
        image_name = image_filename.split('/')[-1].split('.')[0]
        subspace_location = f'../data/subspaces/goi/{image_name}.npy'
        np.save(subspace_location, word_subspace)


if __name__ == '__main__':
    # keyed_vectors = KeyedVectors.load_word2vec_format(w2v_file, limit=vocab_limit, binary=True)
    # space = Space(keyed_vectors)
    # captioner = Captioner(checkpoint_path, word_map_path, device)
    # summarizer = CaptionSummarizer(stop_words=stop_words)
    # image_encoder = ImageEncoder(captioner=captioner, summarizer=summarizer, space=space)
    # generate_subspaces_from_images(images_folder = google_open_images_folder)
    news_summarizer = NewsSummarizer(stop_words=stop_words)
    full_text = \
    "Shocking CCTV footage released by Manchester police shows \
    the moment the man wielding a large-bladed knife is tackled \
    to the ground by armed officers. \
    At about 11 pm on Tuesday, CCTV operators spotted a man \
    waving the butcher’s knife around the Piccadilly Garden’s \
    area of Manchester and informed the police. \
    The man can be seen struggling to stand and interacts with \
    terrified members of the public, as he continues to wave \
    the knife around. \
    A 55-year-old man has been arrested on \
    suspicion of affray and remains in police custody for questioning."
    top_sents = 1
    keywords = set()
    while not keywords:
        keywords_weighed: Counter = news_summarizer.get_keywords(full_text, top=top_sents)
        keywords: set = set(keywords_weighed.keys())
        top_sents += 1
    logger.info(keywords_weighed)
