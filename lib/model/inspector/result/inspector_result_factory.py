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
            _sim_mean = sim_mean(similarities) if len(similarities) > 0 else 0
            begin = True

            column_names = self.__column_names.copy()
            if k == 1:
                column_names.pop(1)

            for wmd_score, sample_desc in similarities:
                row = [wmd_score, remove_pre_post_fix(sample_desc)]

                if not begin:
                    rows_prefix = ['', '', '']
                    rows_postfix = []
                else:
                    begin = False
                    rows_prefix = [pred_desc.strip(), pred_score, _sim_mean]
                    rows_postfix = []

                complete_row = rows_prefix + row + rows_postfix
                if k == 1:
                    complete_row.pop(1)
                rows.append(complete_row)

        return InspectorResult(image_path, pd.DataFrame(rows, columns=column_names))
