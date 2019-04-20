import os
import re
from collections import defaultdict

from lib.utils.array_utils import column
from lib.utils.file_utils import load_text_from
from lib.utils.word_utils import clean_punctuation


class FlickrDataset:
    def __init__(
            self,
            data_path,
            images_path,
            desc_prefix='',
            desc_postfix='',
            separator=r'#[0-9]',
            clean_desc=False
    ):
        self.__separator_pattern = re.compile(separator)
        self.__desc_prefix = desc_prefix
        self.__desc_postfix = desc_postfix
        data = load_text_from(data_path)
        samples = defaultdict(lambda: [])
        self.__max_desc_len = 0

        for index, line in enumerate(data.split('\n')):
            # Exclude headers and invalid lines
            if index == 0 or len(line) < 2:
                continue

            image_path, desc = self.__create_sample(
                line,
                images_path,
                desc_prefix,
                desc_postfix,
                clean_desc
            )

            desc_len = self.__desc_len(desc)
            if desc_len > self.__max_desc_len:
                self.__max_desc_len = desc_len
                max_len_desc = desc

            samples[image_path].append(desc)

        print(f'Max len desc: {max_len_desc}')
        self.__samples = samples

    def __desc_len(self, desc):
        return len(desc) - len(self.__desc_prefix) - len(self.__desc_postfix)

    def max_desc_len(self):
        return self.__max_desc_len

    def samples(self, col=None):
        samples = list(self.__samples.items())
        return samples if col is None else column(samples, col)

    def words_occurs(self):
        words = defaultdict(lambda: 0)
        for descs in self.__samples.values():
            for desc in descs:
                for word in desc.split(' '):
                    if word not in [self.__desc_prefix, self.__desc_prefix]:
                        words[word] = words[word] + 1
        return words

    def words_set(self, min_occurs=0):
        words_occurs = self.words_occurs()
        return [word for word in words_occurs.keys() if words_occurs[word] >= min_occurs]

    def __create_sample(self, line, images_path, desc_prefix, desc_postfix, clean_desc):
        tokens = self.__separator_pattern.split(line)
        image_filename, desc = tokens[0], tokens[1:]

        image_path = os.path.join(images_path, image_filename)

        desc = ' '.join(desc)
        if clean_desc:
            desc = clean_punctuation(desc)
        if len(desc_prefix) > 0:
            desc = f'{desc_prefix} {desc}'
        if len(desc_postfix) > 0:
            desc = f'{desc} {desc_postfix}'

        return image_path, desc
