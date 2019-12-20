from gensim.models import KeyedVectors

from space import Space
from captioner import Captioner

from text.text_utils import get_keywords
from text.summarizer import TextSummarizer

from config import news, images
from config import checkpoint_path, word_map_path, device
from config import w2v_file, vocab_limit

import warnings
warnings.filterwarnings('ignore')

class Media:

    def __init__(self, filepath: str):
        self.filepath = filepath
        self.keywords = None
        self.subspace = None
        try:
            self.name = filepath.split('/')[-1].split('.')[0]
        except IndexError as ie:
            print("Couldn't split filepath string. Setting default media name. {}".format(ie))
            self.name = 'media'

    def __repr__(self):
        return self.get_name()

    def __str__(self):
        return self.get_name()

    def set_keywords(self, keywords: set):
        self.keywords = keywords

    def get_keywords(self):
        return self.keywords

    def set_subspace(self, subspace): # type=numpy.ndarray
        self.subspace = subspace

    def get_subspace(self):
        return self.subspace

    def get_name(self):
        return self.name


class Text(Media):

    def __init__(self, filepath: str):
        super(Text, self).__init__(filepath)

    def full_text(self):
        with open(self.filepath, 'r') as f:
            full_text = f.read()
        return full_text


class Image(Media):

    def __init__(self, filepath: str):
        super(Image, self).__init__(filepath)
        self.captions = None

    def set_captions(self, captions):
        self.captions = captions

    def get_captions(self):
        return self.captions

    def display(self):
        """ How should the image should be displayed? """
        pass


if __name__ == "__main__":
    pass
