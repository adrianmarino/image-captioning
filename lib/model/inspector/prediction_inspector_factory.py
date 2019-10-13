from lib.model.inspector.prediction_inspector import PredictionInspector
from lib.model.inspector.result.inspector_result_factory import InspectorResultFactory
from lib.model.search.search_factory import SearchFactory


class PredictionInspectorFactory:

    @staticmethod
    def create(search, image_features, descriptions):
        return PredictionInspector(
            search,
            image_features,
            InspectorResultFactory(descriptions)
        )

    @staticmethod
    def create_inspector(
            model,
            word_to_index,
            index_to_word,
            image_features,
            descriptions,
            max_seq_len,
            k,
            verbose=True
    ):
        return PredictionInspectorFactory.create(
            search=SearchFactory.beam_search(
                model=model,
                word_to_index=word_to_index,
                index_to_word=index_to_word,
                max_seq_len=max_seq_len,
                k=k,
                verbose=verbose
            ),
            image_features=image_features,
            descriptions=descriptions
        )

