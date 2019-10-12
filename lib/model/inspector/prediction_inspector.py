class PredictionInspector:
    def __init__(self, search, image_features, result_factory):
        self.__search = search
        self.__result_factory = result_factory
        self.__image_features = image_features

    def inspect(self, sample, show=True, pred_desc_count=1):
        predicted_descriptions = self.__predict(sample.path)

        result = self.__result_factory.create(sample, predicted_descriptions, pred_desc_count)
        if show:
            result.show()
        return result

    def __predict(self, image_path):
        return self.__search.perform(self.__image_features[image_path])
