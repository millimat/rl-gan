import tensorflow as tf
from tensorforce.core.networks import Layer

# Cast the output of the previous layer to a desired type.
class Cast(Layer):
    def __init__(self, dtype=None, scope='cast', summary_labels=None):
        super(Cast, self).__init__(scope=scope, summary_labels=summary_labels)
        self.dtype = dtype

    def tf_apply(self, x):
        # Cast to desired type
        if self.dtype is None:
            return tf.identity(x)
        else:
            return tf.cast(x, self.dtype)
