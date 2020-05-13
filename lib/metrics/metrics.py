import numpy as np


def accuracy(y_true, y_pred):
    y_pred = np.cast[np.int32](np.argmax(y_pred, axis=-1))
    y_true = np.cast[np.int32](y_true)
    return np.mean(y_pred == y_true)


def perplexity(y_true, y_pred):
    # y_pred = np.sum(np.eye(y_pred.shape[-1])[np.argmax(y_pred, -1)] * y_pred, -1)
    y_true = np.expand_dims(y_true, axis=-1)
    return np.exp(- np.sum(np.log(y_pred + 0.0001) * y_true) / np.cast[np.float32](y_true.shape[0]))
