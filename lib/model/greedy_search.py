from keras.preprocessing.sequence import pad_sequences
import numpy as np


class GreedySearch:
    def __init__(self,
                 model,
                 word_to_index,
                 index_to_word,
                 seq_prefix,
                 seq_postfix,
                 max_desc_len
                 ):
        self.__model = model
        self.__word_to_index = word_to_index
        self.__index_to_word = index_to_word
        self.__seq_prefix = seq_prefix
        self.__seq_postfix = seq_postfix
        self.__max_desc_len = max_desc_len

    def __to_sequence(self, value):
        return [self.__word_to_index[word] for word in value.split() if word in self.__word_to_index]

    def __remove_begin_end_seq(self, text):
        text = text.split()
        text = text[1:-1]
        text = ' '.join(text)
        return text

    def perform(self, image_feature):
        image_feature = image_feature.reshape((1, 2048))
        description = self.__seq_prefix
        for i in range(self.__max_desc_len):
            sequence = self.__to_sequence(description)
            sequence = pad_sequences([sequence], maxlen=self.__max_desc_len)

            yhat = self.__model.predict([image_feature, sequence], verbose=0)
            yhat = np.argmax(yhat)

            word = self.__index_to_word[yhat]

            description += f' {word}'
            if word == self.__seq_postfix:
                break

        return self.__remove_begin_end_seq(description)
