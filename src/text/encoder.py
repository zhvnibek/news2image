from collections import Counter
from common.basic_encoder import BasicEncoder
from common.space import Space
from text.summarizer import NewsSummarizer
from config import stop_words

class TextEncoder(BasicEncoder):

    def __init__(self, space: Space):
        self.summarizer = NewsSummarizer(stop_words=stop_words)
        self.space = space

    def create_subspace(self, full_text: str, subspace_dim: int = 5):
        """Returns a word subspace """
        keywords: Counter = self._get_keywords(full_text)
        if keywords:
            return self.space.create_subspace(keywords, dims=subspace_dim)

    def _get_keywords(self, full_text: str, limit: int = 7) -> Counter:
        top_sents = 1
        keywords = Counter()
        while not keywords:
            keywords: Counter = self.summarizer.get_keywords(text=full_text, top=top_sents)
            top_sents += 1
        return Counter(dict(keywords.most_common(limit)))

    def __repr__(self) -> str:
        return 'Text Encoder'
