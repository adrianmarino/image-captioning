import keras.backend as K

def rmse(y_true, y_pred): return root_mean_squared_error(y_true, y_pred)

def root_mean_squared_error(y_true, y_pred): return K.sqrt(K.mean(K.square(y_pred - y_true)))

def root_mean_squared_error(y_true, y_pred): return K.sqrt(K.mean(K.square(y_pred - y_true)))

class MetricMeterBuilder:
    def __init__(self, logs): self.logs = logs

    def build(self, metric, value_format='%.6f'):
        value = self.__current_value(metric)
        diff = self.__diff_value(metric)
        count = self.__count(metric)
        return f'{self.label(metric)} ({count}): {value_format % float(value)} ({diff_indicator(diff)})'

    @staticmethod
    def label(value): return value.replace('_', ' ').capitalize()

    def __current_value(self, metric): return self.logs[metric][-1]

    def __count(self, metric): return len(self.logs[metric])

    def __previous_value(self, metric): return self.logs[metric][self.__previous_index(metric)]

    def __previous_index(self, metric): return -2 if len(self.logs[metric]) >= 2 else -1

    def __diff_value(self,  metric): return self.__current_value(metric) - self.__previous_value(metric)


def diff_indicator(value, value_format='%.8f'):
    if value < 0:
        return f'⟱ {value_format % abs(value)}'
    elif value > 0:
        return f'⟰ {value_format % abs(value)}'
    elif value == 0:
        return '='