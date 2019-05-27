import numpy as np


class GloveWordEmbeddingGenerator:
    def __init__(self, path, vector_dim):
        self.__vector_dim = vector_dim
        self.__embedding = {}

        with open(path, encoding='utf-8') as file:
            for line in file:
                values = line.split()
                word = values[0]
                begin_coef_index = self.__begin_coef_index(len(values[1:]))
                coefs = values[begin_coef_index:]

                if len(coefs) != self.__vector_dim:
                    print(f'Skip invalid vector!. Word: "{word}", Coefs: {len(coefs)}, Total: {len(values)}.')
                    continue

                self.__embedding[word] = np.asarray(coefs, dtype='float32')

    def generate(self, word_to_index):
        embedding_matrix = np.zeros((len(word_to_index), self.__vector_dim))

        for word, index in word_to_index.items():
            if word in self.__embedding:
                embedding_matrix[index] = self.__embedding[word]

        return embedding_matrix

    def __begin_coef_index(self, original_size):
        if original_size > self.__vector_dim:
            return original_size - self.__vector_dim + 1

        return 1
