from collections import Counter
from common.basic_encoder import BasicEncoder
from common.space import Space
from text.summarizer import NewsSummarizer


class TextEncoder(BasicEncoder):

    def __init__(self, summarizer: NewsSummarizer, space: Space):
        self.summarizer = summarizer
        self.space = space

    def create_subspace(self, full_text: str, subspace_dim: int = 5):
        """Returns a word subspace """
        keywords: Counter = self._get_keywords(full_text)
        if keywords:
            word_subspace = self.space.create_subspace(keywords, dims=subspace_dim)
            assert(word_subspace.shape==(300, subspace_dim))
        return word_subspace

    def _get_keywords(self, full_text: str) -> Counter:
        keywords = set()
        return Counter(keywords)


    def __str__(self) -> str:
        return 'Text Encoder'
