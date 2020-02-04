# news2meme - An Automated Image Generator from News

### Import modules
```
import os
import numpy as np
from random import randint
from gensim.models import KeyedVectors

from media import Media, Text, Image
from media import create_image_object, create_text_object
from space import Space
from predictor import Model

from config import news, images, image_subspaces, images_folder
from config import w2v_file, vocab_limit
```
### Create a Words Space
```
  """ Load the Keyed Vectors """
  keyed_vectors = KeyedVectors.load_word2vec_format(w2v_file, limit=vocab_limit, binary=True)
  """ Create a Words Space """
  space = Space(keyed_vectors)
```  
### Create an Image Prediction model and set the pre-learned image subspaces
```  
  model = Model(space=space)
  model.set_image_subspaces(image_subspaces)
  
 ```  
 ### Create a Text object from a text file
```
k = randint(0, len(news)+1)
text: Text = create_text_object(news_filename=news[k], space=space)
```

### Check the filename, full text, and its keywords
```
name = text.get_name()
ft = text.full_text()
kw = text.get_keywords()
print("Name: {}\nFull Text:\n {}\nKeywords:\n {}".format(name, ft, kw))
```
### Predict some similar images given the text object
```
baseline_preds = baseline_model.predict(text, count=5)
preds = model.predict(text, count=5)
```
### See the results
```
for image_name, sim in preds:
    image_filename = os.path.join(images_folder, image_name + '.jpg')
    display(ImageOpener(image_filename))
    print(sim)
```
### Image summarization: Textrank 
  https://towardsdatascience.com/understand-text-summarization-and-create-your-own-summarizer-in-python-b26a9f09fc70

## Authors
* **Zhanibek Darimbekov** - *Computer Science, Nazarbayev University* - [GitHub](https://github.com/zhvnibek)
* **Aslan Ubingazhibov** - *Computer Science, Nazarbayev University* - [GitHub](https://github.com/Ubinazhip)
* **Zarina Serikbulatova** - *Computer Science, Nazarbayev University*
