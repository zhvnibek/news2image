from typing_extensions import TypedDict
from gensim.models import KeyedVectors
from src.text.summarizer import NewsSummarizer
from src.recommender import Recommender, Space
from src.config import Word2VecConfig, ImageConfig
from src.tg_bot.merger import make_collage


class Response(TypedDict):
    summary: str
    keywords: str
    collage_loc: str


class Resolver:

    def __init__(self):
        keyed_vectors = KeyedVectors.load_word2vec_format(fname=Word2VecConfig.get_word_vectors_filename(),
                                                          limit=Word2VecConfig.get_vocab_size(),
                                                          binary=True)
        space = Space(keyed_vectors)
        self.recommender = Recommender(space=space)
        self.recommender.set_image_subspaces(path=ImageConfig.get_image_subspaces_folder())
        self.summarizer = NewsSummarizer()
        self.images_folder = ImageConfig.get_images_folder()

    def resolve(self, text: str, options=None) -> Response:
        formatted_summary: str = f"Summary:\n{self.summarizer.generate_summary(text=text)}"
        formatted_keywords: str = f'Keywords:\n{", ".join(set(self.summarizer.get_keywords(text=text)))}'
        collage_loc: str = self._get_collage_loc(text=text)
        return Response(
            summary=formatted_summary,
            keywords=formatted_keywords,
            collage_loc=collage_loc
        )

    def _get_collage_loc(self, text: str):
        images_rec: list = self.recommender.predict(text=text, count=4)
        _ids = [img[0] for img in images_rec]
        return make_collage(image_ids=_ids)
