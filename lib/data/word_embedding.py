import abc
import sys
import types
import warnings

import numpy as np
import tensorflow as tf
import tensorflow_hub as hub
import tqdm

from lib.utils.file_utils import get_num_lines

warnings.filterwarnings('ignore')

if type(tf.contrib) != types.ModuleType:  # if it is LazyLoader
    tf.contrib._warning = None


class WordEmbedding(abc.ABC):
    @abc.abstractmethod
    def encode(self, index_to_word, vocabulary_size):
        pass

    @staticmethod
    def create(name, **kwargs):
        if name.startswith('glove'):
            return GloveWordEmbedding(kwargs['path'], kwargs['vector_dim'])
        elif 'elmo' == name:
            return ElmoWordEmbedding(kwargs['path'])
        else:
            raise Exception(f'Does not exist a word embedding implementation with {name} name!')


@WordEmbedding.register
class ElmoWordEmbedding:
    def __init__(self, path, trainable=False):
        self.__embed = hub.Module(path, trainable=False)

    def __encode(self, words):
        return self.__embed(tf.squeeze(tf.cast(words, tf.string)), signature="default", as_dict=True)["default"]

    def encode(self, word_to_index, vocabulary_size):
        words = [word for word, _ in list(word_to_index.items())]
        with tf.Session() as session:
            session.run([tf.global_variables_initializer(), tf.tables_initializer()])
            embedding_matrix = session.run(self.__encode([words]))

            # Add a zeros vector to first position to begin index from 1 to n.
            first_vector = np.zeros(embedding_matrix.shape[1])
            embedding_matrix = np.vstack([first_vector, embedding_matrix])

            assert vocabulary_size == embedding_matrix.shape[0], f'Invalid embedding_matrix size = {embedding_matrix.shape[0]}'
            return embedding_matrix


@WordEmbedding.register
class GloveWordEmbedding:
    def __init__(self, path, vector_dim, verbose=False):
        self.__vector_dim = vector_dim
        self.__path = path
        self.__verbose = verbose

    def __begin_coef_index(self, original_size):
        return original_size - self.__vector_dim + 1 if original_size > self.__vector_dim else 1

    def __encode(self, word_to_index):
        num_lines = get_num_lines(self.__path)

        with open(self.__path, encoding='utf-8') as file:
            with tqdm.tqdm(total=len(word_to_index), file=sys.stdout) as pbar:
                found_words = 0
                lines_count = 0
                line = file.readline()
                while line and found_words < len(word_to_index):
                    lines_count += 1
                    word, coefficients = self.__word_coefficients(line)

                    if len(coefficients) != self.__vector_dim:
                        if self.__verbose:
                            pbar.write(
                                f'Warn: Skip invalid coefs len!. Word:"{word}", Coefs:{len(coefficients)}, Tot:{len(line.split())}.')
                        continue

                    if word in word_to_index:
                        found_words += 1
                        pbar.update(1)
                        yield (word_to_index[word], coefficients)

                    if pbar.n % 30:
                        pbar.set_description(f'Processing "{word}"')

                    line = file.readline()

                if self.__verbose and found_words < len(word_to_index):
                    pbar.write("Warn: Not found all words!")

    def encode(self, word_to_index, vocabulary_size):
        embedding_matrix = np.zeros((vocabulary_size, self.__vector_dim))

        for index, vector in list(self.__encode(word_to_index)):
            embedding_matrix[index] = vector

        return embedding_matrix

    def __word_coefficients(self, line):
        values = line.split()
        begin_coef_index = self.__begin_coef_index(len(values[1:]))
        return values[0], values[begin_coef_index:]
