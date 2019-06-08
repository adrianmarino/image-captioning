class PredictionInspector:
    def __init__(self, search, image_features, result_factory):
        self.__search = search
        self.__result_factory = result_factory
        self.__image_features = image_features

    def inspect(self, sample, results=3):
        predicted_descriptions = self.__predict(sample[0])
        return self.__result_factory.create(sample, predicted_descriptions, results)

    def __predict(self, image_path):
        return self.__search.perform(self.__image_features[image_path])
