import numpy as np


class GloveWordEmbeddingGenerator:
    def __init__(self, path, vector_dim):
        self.__vector_dim = vector_dim
        self.__embedding = {}

        with open(path, encoding='utf-8') as file:
            for line in file:
                values = line.split()
                word, coefs = values[0], np.asarray(values[1:], dtype='float32')
                self.__embedding[word] = coefs

    def generate(self, word_to_index):
        embedding_matrix = np.zeros((len(word_to_index), self.__vector_dim))

        for word, index in word_to_index.items():
            if word in self.__embedding:
                embedding_matrix[index] =self.__embedding[word]

        return embedding_matrix
