import sys
import os
import numpy as np
from gensim.models import KeyedVectors
from text.encoder import TextEncoder
from common.space import Space
from config import get_logger

sys.path.append(os.path.join(sys.path[0],'image','captioning')) # add models.py

logger = get_logger('recommender')

class Recommender:

    def __init__(self, space: Space):
        self.space = space
        self.text_encoder = TextEncoder(space=self.space)
        self.image_subspaces_loc = None

    def set_image_subspaces(self, path: str):
        self.image_subspaces_loc = path

    def predict(self, text: str, count: int = 5):
        sims = self.compute_similarities(text)
        return sorted(sims, key=lambda x: x[1], reverse=True)[:count]

#     def display_predictions(self, count: int=5):
#         preds = self.predict(text, count)
#         pass

    # Search the most similar word subspace from images
    def compute_similarities(self, text: str):
        sub_txt = self.text_encoder.create_subspace(full_text=text)
        logger.info(sub_txt.shape)
        sims = []
        if self.image_subspaces_loc is not None:
            for im in os.listdir(self.image_subspaces_loc):
                sub_img = np.load(os.path.join(self.image_subspaces_loc, im))
                sim = self.space.subspaces_similarity(sub_txt, sub_img)
                sims.append((im.split('.')[0], sim))
        return sims

    def __str__(self):
        return "Image Recommender Model"
