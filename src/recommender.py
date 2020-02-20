import sys
import os
import numpy as np
from src.text.encoder import TextEncoder
from src.common.space import Space
from src.config import get_logger

sys.path.append(os.path.join(sys.path[0], 'image', 'captioning'))  # add models.py

logger = get_logger('recommender')


class Recommender:

    def __init__(self, space: Space):
        self.space = space
        self.text_encoder = TextEncoder(space=self.space)
        self._image_subspaces_loc = None

    def set_image_subspaces(self, path: str) -> None:
        self._image_subspaces_loc = path

    def predict(self, text: str, count: int = 5) -> list:
        sims = self.compute_similarities(text)
        return sorted(sims, key=lambda x: x[1], reverse=True)[:count]

    def compute_similarities(self, text: str) -> list:
        sub_txt = self.text_encoder.create_subspace(text=text)
        logger.info(sub_txt.shape)
        sims = []
        if self._image_subspaces_loc is not None:
            for im in os.listdir(self._image_subspaces_loc):
                sub_img = np.load(os.path.join(self._image_subspaces_loc, im))
                sim = self.space.subspaces_similarity(sub_txt, sub_img)
                sims.append((im.split('.')[0], sim))
        return sims


if __name__ == '__main__':
    from gensim.models import KeyedVectors
    from src.config import Word2VecConfig, ImageConfig

    t_keyed_vectors = KeyedVectors.load_word2vec_format(fname=Word2VecConfig.get_word_vectors_filename(),
                                                        limit=Word2VecConfig.get_vocab_size(),
                                                        binary=True)
    t_space = Space(t_keyed_vectors)
    t_recommender = Recommender(space=t_space)
    t_recommender.set_image_subspaces(path=ImageConfig.get_image_subspaces_folder())
    t_text = \
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
    preds: list = t_recommender.predict(text=t_text)
    logger.info(preds)
