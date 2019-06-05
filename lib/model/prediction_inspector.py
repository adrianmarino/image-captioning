import pandas as pd
from IPython.display import display

from lib.similarity.sim_utils import SimilarityMeter, mean
from lib.utils.plot_utils import show_img
from lib.utils.word_utils import remove_pre_post_fix

pd.set_option('display.max_colwidth', 400)


class PredictionInspector:
    def __init__(self, search, image_features, descriptions, image_width=500):
        self.__search = search
        self.__image_features = image_features
        self.__image_width = image_width
        self.__similarity_meter = SimilarityMeter(descriptions)

    def __predict(self, image_path):
        return self.__search.perform(self.__image_features[image_path])

    def inspect(self, sample, k=3):
        image_path, sample_descriptions = sample
        predicted_descriptions = self.__predict(image_path)

        show_img(sample[0], image_width=self.__image_width)

        rows = []
        for pred_desc, pred_score in predicted_descriptions[:k]:
            similarities = self.__similarity_meter.measure(pred_desc, sample_descriptions)
            mean_sim = mean(similarities)

            rows_group = [f'{pred_score:2.10f}', pred_desc.strip(), f'{mean_sim:2.10f}']
            index = 0
            for wmd_score, sample_desc in similarities:
                row_postfix = [f'{wmd_score:2.10f}', remove_pre_post_fix(sample_desc)]
                if index > 0:
                    rows_group = ['', '', '']
                else:
                    index += 1
                rows.append(rows_group + row_postfix)

        display(
            pd.DataFrame(
                rows,
                columns=[
                    'Predicted Score (< best)',
                    'Predicted Description',
                    'WMD Mean',
                    'WMD (> best)',
                    'Sample Description'
                ]
            )
        )
