import numpy as np


class GreedySearchStrategy:
    def __init__(self,
                 model,
                 sequencer,
                 index_to_word,
                 seq_prefix,
                 seq_postfix
                 ):
        self.__model = model
        self.__sequencer = sequencer
        self.__index_to_word = index_to_word
        self.__seq_prefix = seq_prefix
        self.__seq_postfix = seq_postfix

    def __remove_begin_end_seq(self, text):
        text = text.split()
        text = text[1:-1]
        text = ' '.join(text)
        return text

    def perform(self, image_feature):
        image_feature = image_feature.reshape((1, 2048))
        phase = self.__seq_prefix

        for seq_index in range(self.max_seq_len()):
            sequence = self.__sequencer.to_seq(phase)

            prob_dist = self.__predict(image_feature, sequence)
            word_position = np.argmax(prob_dist)
            word = self.__index_to_word[word_position]

            phase += f' {word}'
            if word == self.__seq_postfix:
                break

        return self.__remove_begin_end_seq(phase)

    def __predict(self, image_feature, sequence):
        return self.__model.__predict([image_feature, sequence], verbose=0)

    def max_seq_len(self):
        return self.__sequencer.max_seg_len
