import os
import re
from collections import defaultdict

from lib.data.sample import Sample
from lib.utils.file_utils import load_from
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
        data = load_from(data_path)
        samples = {}
        self.__max_desc_len = 0
        max_len_desc = ''

        for index, line in enumerate(data.split('\n')):
            # Exclude headers and invalid lines
            if len(line) < 2:
                continue

            image_filename, image_path, desc = self.__create_sample(
                line,
                images_path,
                desc_prefix,
                desc_postfix,
                clean_desc
            )

            desc_len = len(desc.split())
            if desc_len > self.__max_desc_len:
                self.__max_desc_len = desc_len
                max_len_desc = desc

            if image_filename in samples:
                sample = samples[image_filename]
            else:
                sample = Sample(image_filename, image_path)
                samples[image_filename] = sample

            sample.add_desc(desc)

        print(f'Max len() desc: {max_len_desc}')
        self.__samples = samples

    def max_desc_len(self):
        return self.__max_desc_len

    def samples(self):
        return self.__samples

    def words_occurs(self):
        words = defaultdict(lambda: 0)
        for sample in self.__samples.values():
            for desc in sample.descriptions:
                for word in desc.split(' '):
                    words[word] = words[word] + 1
        return words

    def words_set(self, min_occurs=0):
        words_occurs = self.words_occurs()
        return [word for word in words_occurs.keys() if words_occurs[word] >= min_occurs]

    def descriptions(self): return [desc for sample in self.samples().values() for desc in sample.descriptions]

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

        return image_filename, image_path, desc
