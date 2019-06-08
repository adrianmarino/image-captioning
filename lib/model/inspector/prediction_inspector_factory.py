from lib.model.inspector.prediction_inspector import PredictionInspector
from lib.model.inspector.result.inspector_result_factory import InspectorResultFactory


class PredictionInspectorFactory:

    @staticmethod
    def create(search, image_features, descriptions):
        return PredictionInspector(
            search,
            image_features,
            InspectorResultFactory(descriptions)
        )
