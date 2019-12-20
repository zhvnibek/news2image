from collections import Counter
from common.basic_encoder import BasicEncoder
from common.space import Space
from text.summarizer import CaptionSummarizer
from image.captioning.captioner import Captioner
from config import checkpoint_path, word_map_path, device, stop_words

class ImageEncoder(BasicEncoder):

    def __init__(self, space: Space):
        self.captioner = Captioner(checkpoint_path, word_map_path, device)
        self.summarizer = CaptionSummarizer(stop_words=stop_words)
        self.space = space

    def create_subspace(self, image_filename: str, subspace_dim: int = 5, n_captions: int = 5):
        """Returns a word subspace """
        captions: list = self._get_captions(image_filename, n_captions=n_captions)
        keywords: Counter = self._get_keywords(captions)
        if keywords:
            return self.space.create_subspace(keywords, dims=subspace_dim)

    def _get_captions(self, image_filename: str, n_captions: int = 5) -> list:
        """ Returns the image captions as a list of str """
        return self.captioner.generate_captions(image_filename, n_captions)

    def _get_keywords(self, captions: list) -> Counter:
        keywords = set()
        for caption in captions:
            keywords.update(self.summarizer.get_keywords(text=caption))
        return Counter(keywords)


    def __str__(self) -> str:
        return 'Image Encoder'
