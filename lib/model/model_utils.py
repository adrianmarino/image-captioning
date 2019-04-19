from keras import backend as K


def learning_rate_with_decay(model):
    lr = K.eval(model.optimizer.lr)
    decay = K.eval(model.optimizer.decay)
    iterations = K.eval(model.optimizer.iterations)
    return lr / (1. + decay * iterations)
