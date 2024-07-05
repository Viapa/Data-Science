# 介绍文档：https://blog.csdn.net/ooooocj/article/details/136695008
import tensorflow as tf


# mish激活函数定义
def mish(x):
    return x * tf.tanh(tf.math.softplus(x))


x = tf.random.uniform((10, 3), minval=-5, maxval=5, dtype=tf.float32)
y = mish(x)
print(x)
print(y)

# layer层中使用
tf.keras.utils.get_custom_objects().update({'mish': mish})

inputs = x
out = tf.keras.layers.Dense(64, activation='mish')(inputs)
print(out)