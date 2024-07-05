# 介绍文档：https://blog.csdn.net/ooooocj/article/details/136695008
import tensorflow as tf


# Mish层定义
class Mish(tf.keras.layers.Layer):
    def __init__(self, **kwargs):
        super(Mish, self).__init__(**kwargs)

    def call(self, inputs):
        # mish激活函数: x * tanh(log(1 + exp(x)))
        return inputs * tf.tanh(tf.nn.softplus(inputs))

    def get_config(self):
        return super(Mish, self).get_config()

    def compute_output_shape(self, input_shape):
        return input_shape


# Mish层使用
data = tf.random.normal([100, 10], dtype=tf.float32)
inputs = tf.keras.Input(shape=(10,))
outputs = Mish()(inputs)
model = tf.keras.Model(inputs=inputs, outputs=outputs)
print(data)
print(model(data))
