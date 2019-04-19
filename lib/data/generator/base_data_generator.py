from abc import ABC

import numpy as np
from keras.utils import Sequence

from lib.utils.array_utils import to_chunks


class BaseDataGenerator(Sequence, ABC):
    def __init__(self, samples, batch_size, shuffle=False):
        self._batches = list(to_chunks(samples, batch_size))
        self._shuffle = shuffle

    def __len__(self): return int(np.floor(len(self._batches)))

    def on_epoch_end(self):
        if self._shuffle:
            np.random.shuffle(self._batches)
