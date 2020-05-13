import numpy as np


def accuracy(y_true, y_pred):
    y_pred = np.cast[np.int32](np.argmax(y_pred, axis=-1))
    y_true = np.cast[np.int32](y_true)
    return np.mean(y_pred == y_true)


def perplexity(y_true, y_pred):
    y_true = np.reshape(y_true, [-1, y_pred.shape[1]])
    # mask = np.expand_dims(np.cast[np.float32](np.logical_not(np.equal(y_true, 0))), axis=-1)
    y_true = np.cast[np.float32](np.eye(y_pred.shape[-1])[np.cast[np.int32](y_true)])

    # loss = - (y_true * np.log(y_pred + 0.0001) + (1 - y_true) * np.log(1 - y_pred + 0.0001))
    loss = - y_true * np.log(y_pred + 0.0001)

    # loss *= mask
    loss = np.sum(loss, axis=-1)
    loss = np.mean(loss)

    _perplexity = np.exp(loss)
    return _perplexity
