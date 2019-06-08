import pandas as pd

from lib.model.inspector.result.inspector_result import InspectorResult
from lib.similarity.sim_utils import sim_mean, SimilarityMeter
from lib.utils.word_utils import remove_pre_post_fix


class InspectorResultFactory:

    def __init__(
            self,
            descriptions,
            column_names=[
                'Predicted Description',
                'Score (⟱ best)',
                'WMDSim Mean (⟰ best)',
                'WMDSim (⟰ best)',
                'Sample Description'
            ]
    ):
        self.__column_names = column_names
        self.__similarity_meter = SimilarityMeter(descriptions)

    def create(self, sample, predicted_descriptions, k):
        image_path, sample_descriptions = sample
        predicted_descriptions = sorted(predicted_descriptions, key=lambda tup: tup[1])[:k]
        rows = []

        for pred_desc, pred_score in predicted_descriptions:
            similarities = self.__similarity_meter.measure(pred_desc, sample_descriptions)
            _sim_mean = sim_mean(similarities)
            begin = True

            for wmd_score, sample_desc in similarities:
                row = [f'{wmd_score:2.10f}', remove_pre_post_fix(sample_desc)]

                if not begin:
                    rows_prefix = ['', '', '']
                    rows_postfix = []
                else:
                    begin = False
                    rows_prefix = [pred_desc.strip(), f'{(pred_score):2.10f}', f'{_sim_mean:2.4f}']
                    rows_postfix = []

                rows.append(rows_prefix + row + rows_postfix)

        return InspectorResult(
            image_path,
            pd.DataFrame(rows, columns=self.__column_names)
        )
