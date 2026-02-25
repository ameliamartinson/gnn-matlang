import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()
print(tf.placeholder(tf.float32, shape=(None, 10)))
print('TF 1.x compatibility mode works!')
