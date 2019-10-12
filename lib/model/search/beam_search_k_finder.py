import datetime
import random

import matplotlib.pyplot as plt
import pandas as pd

from lib.model.inspector.prediction_inspector_factory import PredictionInspectorFactory
from lib.utils.pickle_utils import save_obj, load_obj
from lib.utils.plot_utils import display_table


def random_sample(samples):
    index = random.randint(0, len(samples) - 1)
    return samples[index]


def random_samples(samples, count=20):
    results = {}
    while len(results) < count:
        sample = random_sample(samples)
        if sample.id not in results:
            results.update({sample.id: sample})
    return list(results.values())


class BeamSearchKFinder:
    def __init__(
            self,
            model,
            word_to_index,
            index_to_word,
            image_features,
            descriptions,
            max_seq_len
    ):
        self.__model = model
        self.__word_to_index = word_to_index
        self.__index_to_word = index_to_word
        self.__image_features = image_features
        self.__descriptions = descriptions
        self.__max_seq_len = max_seq_len

    def find(self, samples, k_values, verbose=False):
        metrics = []
        for k in k_values:
            start_k_time = datetime.datetime.now()
            inspector = self.__inspector(k, verbose)
            wmd_sim_sum = 0

            for sample in samples:
                result = inspector.inspect(sample, show=False)
                wmd_sim_sum += result.wmd_sim()

            k_time = datetime.datetime.now() - start_k_time
            mean_wmd_sim = wmd_sim_sum / len(samples)
            print(f'K: {k}, Mean WMDSim: {mean_wmd_sim}, Time: {k_time}')
            metrics.append([k, mean_wmd_sim])

        return KMetrics(metrics)

    def __inspector(self, k, verbose=False):
        return PredictionInspectorFactory.create_inspector(
            model=self.__model,
            word_to_index=self.__word_to_index,
            index_to_word=self.__index_to_word,
            image_features=self.__image_features,
            descriptions=self.__descriptions,
            max_seq_len=self.__max_seq_len,
            k=k,
            verbose=verbose
        )


class KMetrics:
    def __init__(self, metrics, path='./metrics'):
        self.table = pd.DataFrame(metrics, columns=['K', 'WMDSim'])
        self.__sorted_table = self.table.sort_values(by=['WMDSim'], ascending=False)
        self.__path = path

    def show_table(self):
        display_table(self.__sorted_table)

    def show_graphs(self):
        plt.figure(figsize=(20, 4))
        plt.xticks(self.table['K'].values)
        plt.step(self.table['K'].values, self.table['WMDSim'].values)
        self.table.plot(kind='line', x='K', y='WMDSim', color='red', figsize=(20, 4))
        plt.xticks(self.table['K'].values)

    def show(self):
        print(f'Best K: {self.best_k()}\n')
        self.show_table()
        self.show_graphs()

    def best_k(self):
        return self.__sorted_table['K'].values[0]

    def worst_k(self):
        return self.__sorted_table['K'].values[-1]

    def save(self):
        save_obj(self.__path, self.table)

    def load(self):
        self.table = load_obj(self.__path)
