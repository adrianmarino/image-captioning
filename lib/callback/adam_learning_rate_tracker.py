from keras.callbacks import Callback
from lib.model.model_utils import learning_rate_with_decay
from lib.utils.dict_utils import dist_values_append

from lib.model.metrics import MetricMeterBuilder


class AdamLearningRateTracker(Callback):
    def __init__(
            self,
            evaluate_interval=100,
            metric_name='learning_rate',
            prefix=''
    ):
        super().__init__()
        self.evaluate_interval = evaluate_interval
        self.metric_name = metric_name
        self.logs = {}
        self.prefix = prefix

    def on_batch_end(self, batch, logs=None):
        if batch % self.evaluate_interval == 0:
            dist_values_append(self.logs, 'learning_rate', learning_rate_with_decay(self.model))
            output = MetricMeterBuilder(self.logs).build(metric=self.metric_name, value_format='%.15f')
            print(f'\n\n{self.prefix}{output}')
