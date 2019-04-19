import matplotlib.pyplot as plt
from IPython.display import clear_output
from dask.tests.test_base import np
from keras.callbacks import Callback
from lib.model.metrics import MetricMeterBuilder
import warnings


def warn(*args, **kwargs):
    pass


warnings.warn = warn


class MetricsPlotter(Callback):
    def __init__(self,
                 validation_generator=None,
                 validation_data=None,
                 plot_interval=10,
                 evaluate_interval=50,
                 batch_size=32
                 ):
        super().__init__()
        self.__plot_interval = plot_interval
        self.__evaluate_interval = evaluate_interval
        self.validation_generator = validation_generator
        self.validation_data = validation_data
        self.batch_size = batch_size
        self.i = 0
        self.val_bach_index = 0
        self.x = []
        self.logs = []
        self.metrics_values = {}
        self.val_metrics_values = {}

    def on_train_begin(self, logs={}):
        for metric in self.model.metrics_names:
            self.metrics_values[metric] = []
            self.val_metrics_values[metric] = []

    def on_batch_end(self, batch, logs={}):
        if batch % self.__plot_interval == 0 and len(self.logs) > 1:
            self.__update_metric_graphs()

        if batch % self.__evaluate_interval == 0:
            self.i += 1
            self.logs.append(logs)
            self.x.append(self.i)

            score = self.__evaluate_model()
            output = self.__build_metric_meters(logs, score)
            if self.i > 1:
                self.__print_meters(output)
            print('\nContinue model train:')

        self.__update_batch_index()


    def __update_batch_index(self):
        if self.val_bach_index < len(self.get_validation_data()):
            self.val_bach_index += 1
        else:
            self.val_bach_index = 0

    def __print_meters(self, output):
        print('\nValidation:')
        [print(f'  - {line}') for line in output]

    def __build_metric_meters(self, logs, score):
        output = []
        meter_builder = MetricMeterBuilder(self.val_metrics_values)
        for index, metric in enumerate(self.model.metrics_names):
            self.metrics_values[metric].append(logs.get(metric))
            self.val_metrics_values[metric].append(score[index])
            output.append(meter_builder.build(metric))
        return output

    def __evaluate_model(self):
        print(f'\n\nEvaluate model (Each {self.__evaluate_interval } steps):')
        val_features, val_labels = self.get_validation_data()
        score = self.model.evaluate(
            val_features,
            val_labels,
            batch_size=self.batch_size,
            verbose=1
        )
        score = np.array(score).flatten()
        return score

    def __update_metric_graphs(self):
        clear_output(wait=True)
        f, axes = plt.subplots(1, len(self.model.metrics_names), figsize=(30, 8))
        for index, metric in enumerate(self.model.metrics_names):
            if len(self.model.metrics_names) > 1:
                axis = axes[index]
            else:
                axis = axes

            axis.plot(self.x, self.metrics_values[metric], label=metric)
            axis.plot(self.x, self.val_metrics_values[metric], label=f'val_{metric}')
            axis.legend()
        plt.show()

    def validation_data_len(self):
        return len(self.validation_generator) if self.validation_data is None else len(self.validation_data[0])

    def get_validation_data(self):
        if self.validation_data is None:
            return self.validation_generator[self.val_bach_index]

        return self.validation_data[0], self.validation_data[1]
