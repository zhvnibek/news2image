import os
from typing_extensions import TypedDict
from gensim.models import KeyedVectors
from src.text.summarizer import NewsSummarizer
from src.image.encoder import ImageEncoder
from src.recommender import Recommender, Space
from src.config import Word2VecConfig, ImageConfig, PostProcessingConfig
from src.tg_bot.merger import make_collage

import sys
sys.path.append("/home/zhanibek/Desktop/Fall '19/Senior Project/news2image/src/image/captioning")  # add models.py


class Response(TypedDict):
    summary: str
    keywords: str
    collage_loc: str
    captions: str


class Resolver:

    def __init__(self):
        keyed_vectors = KeyedVectors.load_word2vec_format(fname=Word2VecConfig.get_word_vectors_filename(),
                                                          limit=Word2VecConfig.get_vocab_size(),
                                                          binary=True)
        space = Space(keyed_vectors=keyed_vectors)
        self.recommender = Recommender(space=space)
        self.recommender.set_image_subspaces(path=ImageConfig.get_image_subspaces_folder())
        self.summarizer = NewsSummarizer()
        self.image_encoder = ImageEncoder(space=space)
        self.images_folder = ImageConfig.get_images_folder()

    @staticmethod
    def get_country_flag_locs(countries: set) -> list:
        """Returns location of the image file"""
        selected_flags = []
        for flag in os.listdir(PostProcessingConfig.get_flags_folder()):
            for country in countries:
                if country.capitalize() in flag:
                    selected_flags.append(os.path.join(PostProcessingConfig.get_flags_folder(), flag))
        return selected_flags

    def resolve(self, text: str, params=None) -> Response:
        formatted_summary: str = f"*Summary:*\n_{self.summarizer.generate_summary(text=text)}_"
        keywords = set(self.summarizer.get_keywords(text=text))
        countries_set: set = PostProcessingConfig.get_country_names()
        countries_set.intersection_update(keywords)
        print(f'Countries: {countries_set}')
        formatted_keywords: str = f'*Keywords:*\n_{", ".join(keywords)}_'
        images_rec: list = self.recommender.predict(text=text, count=4)
        _ids = [img[0] for img in images_rec]
        print(f'Selected image IDs: {_ids}')
        countries: list = Resolver.get_country_flag_locs(countries=countries_set)
        collage_loc = make_collage(image_ids=_ids, countries=countries)
        data: dict = self._get_image_captions_and_keywords(image_ids=_ids)
        captions = ""
        for k, v in data.items():
            captions += f"Image #{k+1}:\n*Сaption:* _{v.get('caption')}_\n*Keywords:* _{', '.join(v.get('keywords', []))}_\n\n"
        return Response(
            summary=formatted_summary,
            keywords=formatted_keywords,
            collage_loc=collage_loc,
            captions=captions
        )

    def _get_image_captions_and_keywords(self, image_ids: list) -> dict:
        d = {}
        image_full_filenames = [os.path.join(self.images_folder, f'{_id}.jpg') for _id in image_ids]
        for i, im in enumerate(image_full_filenames):
            t_captions: list = self.image_encoder._get_captions(image_filename=im)
            print(t_captions)
            d[i] = {}
            if t_captions:
                d[i]['caption'] = t_captions[0]
            t_keywords = self.image_encoder._get_keywords(captions=t_captions)
            print(t_keywords)
            if t_keywords:
                d[i]['keywords'] = list(t_keywords)
        return d

    def _get_collage_loc(self, text: str):
        images_rec: list = self.recommender.predict(text=text, count=4)
        _ids = [img[0] for img in images_rec]
        print(_ids)
        return make_collage(image_ids=_ids)


if __name__ == '__main__':
    from pprint import pprint
    resolver = Resolver()
    t_txt = """Oil prices crashed in Asia on Monday by around 30% in what analysts are calling the start of a price war.
    Top oil exporter Saudi Arabia slashed its oil prices at the weekend after it failed to convince Russia on Friday to back sharp production cuts.
    Oil cartel Opec and its ally Russia had previously worked together on production curbs.
    The benchmark Brent oil futures plunged to a low of $31.02 a barrel on Monday, in volatile energy markets.
    Oil prices have now fallen 30% since Friday, when Opec's 14 members led by Saudi Arabia met with its allies Russia and other non-Opec members.
    They met to discuss how to respond to falling demand caused by the growing spread of the coronavirus."""
    r: Response = resolver.resolve(text=t_txt)
    pprint(r)
    # fl = Resolver.get_country_flag_locs({"China", "Russia"})
    # print(fl)
