import tensorflow as tf
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Dense
from tensorflow.keras.layers import Input
from tensorflow.keras.models import Model


class NormalizationLayer(tf.keras.layers.Layer):
    def __init__(self, minval=0., maxval=1., **kwargs):
        super(NormalizationLayer, self).__init__(**kwargs)

        self.minval = minval
        self.maxval = maxval

    def build(self, input_shape):
        super(NormalizationLayer, self).build(input_shape)

    def call(self, inputs):
        return tf.math.sigmoid(inputs) * (self.maxval -
                                          self.minval) + self.minval

    def get_config(self):
        config = super(NormalizationLayer, self).get_config()
        config.update({"minval": self.minval, "maxval": self.maxval})
        return config



class AE(tf.keras.Model):
    def __init__(self,
                 input_shape,
                 latent_dim,
                 arch_string='d512.d256.d128',
                 normalize=None):
        super(AE, self).__init__()
        self._latent_dim = latent_dim
        self._input_shape = input_shape
        self._normalize = normalize
        self.build_encoder(arch_string)
        self.build_decoder(arch_string)

    def encode(self, x, batch_size=256):
        return self.encoder.predict(x, batch_size=batch_size)

    def decode(self, z, batch_size=256):
        return self.decoder.predict(z, batch_size=batch_size)

    def build_encoder(self, arch_string):
        x = Input(shape=self._input_shape)
        y = x
        for s in arch_string.split('.'):
            if s.startswith('d'):
                dim = int(s[1:])
                y = Dense(dim)(y)
                y = Activation('relu')(y)
            else:
                raise NotImplementedError(
                    f"Unknown layer identifier {s[0]} in the Encoder")

        y = Dense(self._latent_dim)(y)

        self.encoder = Model(x, y)

    def build_decoder(self, arch_string):
        x = Input(shape=(self._latent_dim, ))
        y = x
        for s in arch_string.split('.')[::-1]:
            if s.startswith('d'):
                dim = int(s[1:])
                y = Dense(dim)(y)
                y = Activation('relu')(y)
            else:
                raise NotImplementedError(
                    f"Unknown layer identifier {s[0]} in the Decoder")

        y = Dense(self._input_shape[0])(y)
        if self._normalize is not None:
            y = NormalizationLayer(minval=self._normalize[0],
                                   maxval=self._normalize[1],
                                   name='normalize')(y)
        self.decoder = Model(x, y, name="decoder")

    def call(self, x):
        z = self.encoder(x)
        return self.decoder(z)

