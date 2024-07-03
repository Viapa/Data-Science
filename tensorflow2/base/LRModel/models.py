import tensorflow as tf
from tensorflow.keras import models, layers


class MyModel(models.Model):
    def __init__(self, hidden_size, **kwargs):
        super(MyModel, self).__init__()
        self.hidden_size = hidden_size

    def build(self, input_shape):
        self.fc1 = layers.Dense(self.hidden_size, activation='relu')
        self.fc2 = layers.Dense(self.hidden_size // 2, activation='relu')
        self.fc3 = layers.Dense(1, activation='sigmoid')
        self.bn1 = layers.BatchNormalization()
        self.bn2 = layers.BatchNormalization()
        self.dropout = layers.Dropout(0.2)
        super(MyModel, self).build(input_shape)

    def call(self, inputs):
        x = self.fc1(inputs)
        x = self.bn1(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.bn2(x)
        out = self.fc3(x)

        return out
