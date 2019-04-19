from IPython.display import SVG, display
from keras.utils.vis_utils import model_to_dot
import keras.backend as K

from lib.callback.metric_plotter import MetricsPlotter


class ModelWrapper:
    def __init__(self, model):
        self.__model = model

    def lr(self): return K.eval(self.__model.optimizer.lr)

    def set_lr(self, value):
        self.__model.optimizer.lr = value
        return self

    def show(self):
        print(f'\n\nMetrics: {self.metrics_names()}')
        self.__model.summary()
        display(SVG(model_to_dot(
            self.__model,
            show_shapes=True,
            show_layer_names=True
        ).create(prog='dot', format='svg')))

    def fit(self, train_generator, val_generator, epochs, steps_per_epoch, callbacks):
        print(f'LR: {self.lr()}')

        return self.__model.fit_generator(
            generator=train_generator,
            steps_per_epoch=steps_per_epoch,
            epochs=epochs,
            validation_data=val_generator,
            callbacks=callbacks,
            verbose=1,
            use_multiprocessing=True,
            workers=8
        )

    def metrics_names(self): return self.__model.metrics_names