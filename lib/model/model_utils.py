import re

from keras import backend as K

from lib.utils.os_utils import min_file_path_from


def learning_rate_with_decay(model):
    lr = K.eval(model.optimizer.lr)
    decay = K.eval(model.optimizer.decay)
    iterations = K.eval(model.optimizer.iterations)
    return lr / (1. + decay * iterations)


def get_loss_model_weights_path(file_path):
    loss = re.search("(.*)val_loss_(\d+\.\d+|\d+)(.*).h5", file_path).group(2)
    return float(loss)


def get_best_weights_file_from(path):
    path_pattern = f'./{path}/*.h5'
    result = min_file_path_from(path_pattern, get_loss_model_weights_path)
    print(result)
    return None if path_pattern == result else result
