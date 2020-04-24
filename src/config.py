import os
import logging
import torch
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(levelname)s %(message)s')


def get_logger(name: str):
    return logging.getLogger(name)


class StopWordsConfig:
    _stopwords_file = "/home/zhanibek/Desktop/Spring20/Senior Project II/news2image/src/text/utils/stopwords.txt"
    _stopwords = set(stopwords.words('english'))

    @staticmethod
    def get_stopwords(include_from_file: bool = True):
        if include_from_file:
            with open(file=StopWordsConfig._stopwords_file, mode='r') as f:
                _extra = f.read().split('\n')
            StopWordsConfig._stopwords |= set(_extra)
        return StopWordsConfig._stopwords


class TextConfig:
    _texts_folder = "/home/zhanibek/Desktop/Spring20/Senior Project II/news2image/data/texts"
    _texts_subspaces_folder = None

    @staticmethod
    def get_texts_folder() -> str:
        return TextConfig._texts_folder

    @staticmethod
    def get_text_category_folder(category: str = 'news') -> str:
        return os.path.join(TextConfig._texts_folder, category)

    @staticmethod
    def get_text_subspaces_folder() -> str:
        return TextConfig._texts_subspaces_folder


class ImageConfig:
    _images_folder = "/home/zhanibek/Desktop/Spring20/Senior Project II/news2image/data/images/goi5k"
    _image_subspaces_folder = "/home/zhanibek/Desktop/Spring20/Senior Project II/news2image/data/subspaces/goi5k"

    @staticmethod
    def get_images_folder() -> str:
        return ImageConfig._images_folder

    @staticmethod
    def get_image_subspaces_folder() -> str:
        return ImageConfig._image_subspaces_folder


class CaptionerConfig:
    checkpoint_path = "/home/zhanibek/Desktop/Spring20/Senior Project II/news2image/data/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
    word_map_path = "/home/zhanibek/Desktop/Spring20/Senior Project II/news2image/data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


class Word2VecConfig:
    _w2v_file = '~/Downloads/GoogleNews-vectors-negative300.bin.gz'
    _vocab_size = 20000
    _vocab_max_size = 3000000

    @staticmethod
    def get_word_vectors_filename() -> str:
        return Word2VecConfig._w2v_file

    @staticmethod
    def get_vocab_size() -> int:
        return Word2VecConfig._vocab_size

    @staticmethod
    def set_vocab_size(size: int) -> None:
        if Word2VecConfig._vocab_max_size > size > 10000:
            Word2VecConfig._vocab_size = size


class PostProcessingConfig:
    _country_flags_folder = "/home/zhanibek/Desktop/Spring20/Senior Project II/news2image/data/countries/flags_png"
    _country_names_txt = "/home/zhanibek/Desktop/Spring20/Senior Project II/news2image/data/countries/countries.txt"

    @staticmethod
    def get_country_names() -> set:
        with open(file=PostProcessingConfig._country_names_txt, mode='r') as f:
            c = f.read()
        return set(c.split('\n'))

    @staticmethod
    def get_flags_folder() -> str:
        return PostProcessingConfig._country_flags_folder
