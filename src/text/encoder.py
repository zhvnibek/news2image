from collections import Counter
from src.common.basic_encoder import BasicEncoder
from src.common.space import Space
from src.text.summarizer import NewsSummarizer
from src.config import StopWordsConfig


class TextEncoder(BasicEncoder):

    def __init__(self, space: Space):
        super().__init__(space)
        self.summarizer = NewsSummarizer(stopwords=StopWordsConfig.get_stopwords())

    def create_subspace(self, text: str, dim: int = 5):
        """Returns a word subspace """
        keywords: Counter = self._get_keywords(text=text)
        if keywords:
            return self.space.create_subspace(keywords=keywords, dim=dim)

    def _get_keywords(self, text: str, limit: int = 5) -> Counter:
        sentence_limit = 1
        keywords = Counter()
        while not keywords:
            keywords: Counter = self.summarizer.get_keywords(text=text, sentence_limit=sentence_limit, keyword_limit=limit)
            sentence_limit += 1
        # print(f'Encoder. Keywords: {keywords}')
        return keywords


if __name__ == '__main__':
    from gensim.models import KeyedVectors
    from src.config import Word2VecConfig

    t_keyed_vectors = KeyedVectors.load_word2vec_format(fname=Word2VecConfig.get_word_vectors_filename(),
                                                        limit=Word2VecConfig.get_vocab_size(),
                                                        binary=True)
    t_space = Space(t_keyed_vectors)
    text_encoder = TextEncoder(space=t_space)

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
    t_keywords = text_encoder._get_keywords(text=t_text)
    print(t_keywords)