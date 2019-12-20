import numpy as np
import networkx as nx
from nltk import pos_tag
from nltk.cluster.util import cosine_distance
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize, sent_tokenize, wordpunct_tokenize
from collections import Counter


class Summarizer:

    def __init__(self, stop_words: set = set()):
        self.lemmatizer = WordNetLemmatizer()
        self.stop_words: set = stop_words

    def word_counts(self, text: str) -> Counter:

        return Counter(
            [
                self.lemmatizer.lemmatize(token.lower(), pos='v')
                    for token in wordpunct_tokenize(text)
                    if token.isalpha() and token not in self.stop_words
            ]
        )

    def _filter_text(self, text: str, stem=True) -> set:
        def _stem_filter(text: str) -> str:
            tokens = [t.lower() for t in wordpunct_tokenize(text)]
            return " ".join(self.lemmatizer.lemmatize(token, pos='v') for token in tokens)

        def _tagged_filter(text: str) -> set:
            selected = ['CD', 'FW', 'JJ', 'NN', 'NNP', 'NNS', 'NNPS', 'VBG', 'VBZ', 'VBP']
            tokens = [t.lower() for t in word_tokenize(text) if t.isalpha() and t not in self.stop_words]
            tagged_text = pos_tag(tokens)
            joined = " ".join([word[0] for word in tagged_text if word[1] in selected])
            return set([word[0] for word in tagged_text if word[1] in selected])

        return _tagged_filter(_stem_filter(text)) if stem else _tagged_filter(text)

    def get_keywords(self, text: str) -> dict:
        raise NotImplementedError

class CaptionSummarizer(Summarizer):

    def __init__(self, stop_words):
        super().__init__(stop_words=stop_words)

    def get_keywords(self, text: str) -> set:
        """Returns keywords with weights"""
        keywords: set = self._filter_text(text)
        # counter: Counter = self.word_counts(text)
        # return Counter({k: counter[k] for k in keywords})
        return keywords

class NewsSummarizer(Summarizer):

    def __init__(self, stop_words):
        super().__init__(stop_words=stop_words)

    def _sentence_similarity(self, first: str, second: str) -> float:
        words1 = word_tokenize(first)
        words2 = word_tokenize(second)
        all_words = list(set(words1 + words2))
        vector1 = [0] * len(all_words)
        vector2 = [0] * len(all_words)

        for w in words1:
            if w in self.stop_words:
                continue
            vector1[all_words.index(w)] += 1

        for w in words2:
            if w in self.stop_words:
                continue
            vector2[all_words.index(w)] += 1

        return 1 - cosine_distance(vector1, vector2)

    def _build_ssm(self, sentences: list):
        SSM = np.zeros((len(sentences), len(sentences)))

        for idx1 in range(len(sentences)):
            for idx2 in range(len(sentences)):
                if idx1 == idx2: # ignore if both are same sentences
                    continue
                SSM[idx1][idx2] = self._sentence_similarity(sentences[idx1], sentences[idx2])

        for idx in range(len(SSM)):
            SSM[idx] /= SSM[idx].sum()

        return SSM

    def _generate_summary(self, text: str, top: int=1) -> str:
        sentences: list = sent_tokenize(text)
        top = min(top, len(sentences))

        SSM = self._build_ssm(sentences)
        SSG = nx.from_numpy_array(SSM)
        scores = nx.pagerank(SSG)
        ranked_sentences = sorted(((scores[i], s) for i, s in enumerate(sentences)), reverse=True)

        return " ".join([ranked_sentences[i][1] for i in range(top)])

    def get_keywords(self, text: str, top: int=1) -> Counter:
        """Get keywords with weights"""
        summary: str = self._generate_summary(text, top)
        keywords: set = self._filter_text(summary)
        counter: Counter = self.word_counts(text)
        return Counter({k: counter[k] for k in keywords})

if __name__=='__main__':
    with open("stopwords.txt", 'r') as f:
        stop_extra = f.read().split('\n')
    stop_words = set(stopwords.words('english'))
    stop_words |= set(stop_extra)

    full_text = \
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
    ns = NewsSummarizer(stop_words=stop_words)
    top_sents = 1
    keywords = set()
    while not keywords:
        keywords_weighed: Counter = ns.get_keywords(full_text, top=top_sents)
        keywords: set = set(keywords_weighed.keys())
        top_sents += 1

    # cs = CaptionSummarizer(stop_words=stop_words)
    # caption = 'a close up of a broccoli head on a table table'
    # keywords_weighed = cs.get_keywords(text=caption)
    print("Keywords: {}".format(keywords))
    print(keywords_weighed)
