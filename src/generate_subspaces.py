import sys
import os
import numpy as np
from collections import Counter
from gensim.models import KeyedVectors
from common.space import Space
from image.encoder import ImageEncoder
from text.encoder import TextEncoder
from text.summarizer import NewsSummarizer
from config import w2v_file, vocab_limit
from config import google_open_images_folder
from config import get_logger

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
        break

if __name__ == '__main__':
    keyed_vectors = KeyedVectors.load_word2vec_format(w2v_file, limit=vocab_limit, binary=True)
    space = Space(keyed_vectors)
    image_encoder = ImageEncoder(space=space)
    generate_subspaces_from_images(images_folder = google_open_images_folder)

    # text_encoder = TextEncoder(summarizer=news_summarizer, space=space)
