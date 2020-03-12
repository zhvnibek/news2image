import torch
from torchvision import models, transforms
from PIL import Image
from config import images, imagenet_classes
from text.text_utils import get_keywords


class TagExtractor:

    def __init__(self, classifier, imagenet_classes: list):
        self.classifier = classifier
        self.classes = imagenet_classes
        self.transform = transforms.Compose([
            transforms.Resize(256),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(
                mean=[0.485, 0.456, 0.406],
                std=[0.229, 0.224, 0.225]
            )])

    def predict_tags(self, image_filename, count: int = 5):
        img = Image.open(image_filename)
        try:
            img_t = self.transform(img)
        except Exception as e:
            print("Couldn't transform the image. {}".format(e))
            return None
        batch_t = torch.unsqueeze(img_t, 0)
        preds = self.classifier(batch_t)
        _, indices = torch.sort(preds, descending=True)
        #         percentage = torch.nn.functional.softmax(preds, dim=1)[0] * 100
        #         tags_with_probs = [(self.classes[idx], percentage[idx].item()) for idx in indices[0][:count]]
        #         print(tags_with_probs)
        raw_tags = " ".join([self.classes[idx] for idx in indices[0]][:count])
        #         print(raw_tags)
        # cleaned, tagged = prepare_text(raw_tags)
        # keywords = set(cleaned.split())
        keywords = get_keywords(raw_tags)
        return keywords


"""" Pre-trained alexnet image classifier """
# alexnet = models.alexnet(pretrained=True)
# _ = alexnet.eval()

""" Create a Tag Extractor """
# tag_extractor = TagExtractor(classifier=alexnet, imagenet_classes=imagenet_classes)

if __name__ == '__main__':
    """" Pre-trained alexnet image classifier """
    alexnet = models.alexnet(pretrained=True)
    _ = alexnet.eval()

    """ Create a Tag Extractor """
    tag_extractor = TagExtractor(classifier=alexnet, imagenet_classes=imagenet_classes)
    """ Get some image """
    k = torch.randint(0, len(images), ()).item()
    img = images[k]

    """ Predict the tags (keywords) """
    keywords = tag_extractor.predict_tags(image_filename=img, count=7)
    print(keywords)
