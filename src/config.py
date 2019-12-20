import logging
import os
import torch
from nltk.corpus import stopwords

logging.basicConfig(level=logging.INFO, format='%(asctime)s %(name)s: %(levelname)s %(message)s')

def get_logger(name: str):
    return logging.getLogger(name)

""" Stopwords """
with open("text/utils/stopwords.txt", 'r') as f:
    stop_extra = f.read().split('\n')
stop_words = set(stopwords.words('english'))
stop_words |= set(stop_extra)

""" News Texts """
# news_folder = "/home/zhanibek/Desktop/Fall '19/Senior Project/dataset-download/texts"
# news = []
# for category in os.listdir(news_folder):
#     news.extend([os.path.join(news_folder, category, txt) for txt in os.listdir(os.path.join(news_folder, category))])

""" Images """
# images_folder = "/home/zhanibek/Desktop/Fall '19/Senior Project/dataset-download/images"
# images = [os.path.join(images_folder, image_filename) for image_filename in os.listdir(images_folder)]
google_open_images_folder = "/home/zhanibek/Desktop/Fall '19/Senior Project/dataset-download/google_open_images/open_images"
google_open_images = [os.path.join(google_open_images_folder, image_filename) for image_filename in os.listdir(google_open_images_folder)]

""" Captioner """
checkpoint_path = "../data/BEST_checkpoint_coco_5_cap_per_img_5_min_word_freq.pth.tar"
word_map_path = "../data/WORDMAP_coco_5_cap_per_img_5_min_word_freq.json"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

""" Pretrained word vectors """
w2v_file = '~/Downloads/GoogleNews-vectors-negative300.bin.gz'
vocab_limit = 1000000

"""" Image Subspaces """
# image_subspaces = ("/home/zhanibek/Desktop/Fall '19/Senior Project/news2meme/subspaces/image", 0) # from small dataset w/ IC
# image_subspaces_alexnet = ("/home/zhanibek/Desktop/Fall '19/Senior Project/news2meme/alexnet_subspaces", 1) # from small dataset w/ AlexNet
# google_open_subspaces = ("/home/zhanibek/Desktop/Fall '19/Senior Project/news2meme/google_open_subspaces", 2) # from google images w/ IC

""" ImageNet classes """
# with open('imagenet_classes.txt') as f:
#     imagenet_classes = [line.strip() for line in f.readlines()]
