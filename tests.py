import tensorflow as tf

embedding_dimension = 10
negative_samples_size = 10

negative_samples = tf.placeholder(shape=[embedding_dimension, negative_samples_size, None], dtype=tf.float64)
