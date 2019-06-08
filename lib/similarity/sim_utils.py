import os
from time import time
from nltk import word_tokenize
from nltk import download
from nltk.corpus import stopwords
from gensim.models import Word2Vec
from gensim.similarities import WmdSimilarity
from functools import reduce

download('punkt')  # Download data for tokenizer.
download('stopwords')  # Download stopwords list.
stop_words = stopwords.words('english')


def preprocess(doc):
    doc = doc.lower()  # Lower the text.
    doc = word_tokenize(doc)  # Split into words.
    doc = [w for w in doc if not w in stop_words]  # Remove stopwords.
    doc = [w for w in doc if w.isalpha()]  # Remove numbers and punctuation.
    return doc


def normalize(descriptions):
    return [preprocess(desc) for desc in descriptions]


def sim_mean(similarities):
    return reduce(lambda a, b: a + b, [sim[0] for sim in similarities]) / len(similarities)


class SimilarityMeter:
    def __init__(self, texts, embedding_path='world2vec_embedding.model'):
        normalized_texts = normalize(texts)

        if not os.path.isfile(embedding_path):
            self.__model = Word2Vec(normalized_texts, workers=8, size=100)
            self.__model.save(embedding_path)
        else:
            self.__model  = Word2Vec.load(embedding_path)

    def measure(self, of_text, with_texts):
        instance = WmdSimilarity(normalize(with_texts), self.__model, num_best=len(with_texts))
        sim = instance[preprocess(of_text)]
        return [(sim[index][1], with_texts[sim[index][0]]) for index in range(len(sim))]