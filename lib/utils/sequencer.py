from keras.preprocessing.sequence import pad_sequences


class Sequencer:
    def __init__(self, word_to_index, index_to_word, max_seq_len):
        self.__word_to_index = word_to_index
        self.__index_to_word = index_to_word
        self.max_seg_len = max_seq_len

    def to_seq(self, phase):
        return [self.__word_to_index[word] for word in phase.split() if word in self.__word_to_index]

    def to_pad_seq(self, phase):
        return self.pad(self.to_seq(phase))

    def pad(self, sequence):
        return pad_sequences([sequence], maxlen=self.max_seg_len)

    def to_phrase(self, sequence):
        return ' '.join([self.__index_to_word[i] for i in sequence])
