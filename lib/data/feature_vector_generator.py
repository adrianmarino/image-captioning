import cv2
import numpy as np
from keras import Model
from keras.applications.inception_v3 import InceptionV3, preprocess_input
import tqdm


class FeatureVectorGenerator:
    def __init__(self, weights='imagenet', input_shape=(299, 299)):
        self.__input_shape=input_shape
        # Load the inception v3 model
        original_model = InceptionV3(weights=weights)
        # Create a model only with convolution layers of inception model
        self.__model = Model(original_model.input, original_model.layers[-2].output)

    # Function to encode a given image into a vector of size (2048, )
    def generate(self, image_paths):
        for index in tqdm.tqdm(range(len(image_paths))):
            image_path = image_paths[index]
            try:
                # preprocess the image
                model_input = self.__to_model_input(image_path)

                # Get the encoding vector for the image
                feature_vector = self.__model.predict(model_input,  verbose=0)

                # reshape from (1, 2048) to (2048, )
                feature_vector = np.reshape(feature_vector, feature_vector.shape[1])

                yield (image_path, feature_vector)

            except:
                print(f'Error when process {image_path} image!. This was excluded!')

    def __to_model_input(self, img_path):
        img = cv2.imread(img_path, cv2.IMREAD_COLOR)
        img = cv2.resize(img, self.__input_shape)

        # wrap array into another array
        img = np.expand_dims(img, axis=0)

        # Preprocess images using preprocess_input() from inception module
        return preprocess_input(img)
