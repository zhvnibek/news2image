"""
    'Show, Attend, and Tell' Implementation in PyTorch from
    https://github.com/sgrvinod/a-PyTorch-Tutorial-to-Image-Captioning
"""
from image.captioning import ic_model
import torch
import json
import warnings
warnings.filterwarnings("ignore")


class Captioner:

    def __init__(self, checkpoint_path: str, word_map_path: str, device: str = 'cpu'):
        self.checkpoint = torch.load(checkpoint_path, map_location=torch.device(device))
        self.decoder = self.checkpoint['decoder'].to(device)
        self.decoder.eval()
        self.encoder = self.checkpoint['encoder'].to(device)
        self.encoder.eval()
        with open(word_map_path, 'r') as f:
            self.word_map = json.load(f)
        self.rev_word_map = {v: k for k, v in self.word_map.items()}  # ix2word

    def generate_captions(self, image_filename: str, n_captions: int = 5):
        captions = []
        # Encode, decode with attention and beam search
        for i in range(1, n_captions + 1):
            seq, alphas = ic_model.caption_image_beam_search(self.encoder, self.decoder, image_filename, self.word_map, i)
            alphas = torch.FloatTensor(alphas)

            captions.append(ic_model.return_sentence(image_filename, seq, alphas, self.rev_word_map))

        return captions
