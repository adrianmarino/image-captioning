from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical
from numpy import array
from lib.data.generator.base_data_generator import BaseDataGenerator


class DataGenerator(BaseDataGenerator):
    def __init__(
            self,
            samples,
            image_features,
            word_to_index,
            index_to_word,
            max_length,
            vocabulary_size,
            batch_size,
            shuffle=False
    ):
        super().__init__(samples, batch_size, shuffle)
        self.__image_features = image_features
        self.__word_to_index = word_to_index
        self.__index_to_word = index_to_word
        self.__max_length = max_length
        self.__vocabulary_size = vocabulary_size

    def __getitem__(self, index):
        X1, X2, y = list(), list(), list()

        for image_path, image_descriptions in self._batches[index]:
            if image_path not in self.__image_features:
                print(f'Not found image feature for {image_path}. Skip sample!')
                continue

            image_feature = self.__image_features[image_path]

            for image_desc in image_descriptions:
                desc_seq = self.__to_sequence(image_desc)

                # split one sequence into multiple X, y pairs
                for seq_index in range(1, len(desc_seq)):
                    # split into input and output pair
                    in_seq, out_seq = desc_seq[:seq_index], desc_seq[seq_index]

                    X1.append(image_feature)
                    X2.append(self.__pad(in_seq))
                    y.append(self.__vocabulary_one_hot_vector(out_seq))

        return [[array(X1), array(X2)], array(y)]

    def __pad(self, in_seq):
        return pad_sequences([in_seq], maxlen=self.__max_length)[0]

    def __vocabulary_one_hot_vector(self, value):
        return to_categorical([value], num_classes=self.__vocabulary_size)[0]

    def __to_sequence(self, image_desc):
        return [self.__word_to_index[word] for word in image_desc.split(' ') if word in self.__word_to_index]
