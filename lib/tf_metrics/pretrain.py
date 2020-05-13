import tensorflow as tf


def tf_accuracy(y_true, y_pred):
    with tf.name_scope('accuracy'):
        y_pred = tf.argmax(y_pred, axis=-1)
        y_true = tf.cast(y_true, y_pred.dtype)
        return tf.reduce_mean(tf.cast(tf.equal(y_pred, y_true), tf.float32))


def tf_perplexity(y_true, y_pred):
    with tf.name_scope('perplexity'):
        y_true = tf.reshape(y_true, [-1, y_pred.shape[1]])
        y_true = tf.cast(tf.one_hot(tf.cast(y_true, tf.int32), y_pred.shape[-1]), y_pred.dtype)
        return tf.exp(tf.losses.categorical_crossentropy(y_true, y_pred))
