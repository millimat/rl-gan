import tensorflow as tf
from tensorforce.core.preprocessors import Preprocessor

class Cast(Preprocessor):
    # Cast state to a different type.
    # Useful for discrete states since Tensorforce layers only support tf.float32.
    def __init__(self, shape=(), dtype=None, scope='cast', summary_labels=()):
        super(Cast, self).__init__(shape, scope=scope, summary_labels=summary_labels)
        self.dtype = dtype

        def tf_process(self, tensor):
            import pdb; pdb.set_trace()
            # Cast to desired dtype
            if self.dtype is None:
                return tf.identity(tensor)
            else:
                return tf.cast(tensor, self.dtype)
