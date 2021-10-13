import numpy as np

from scriptify import scriptify

from latent_space import AE

import tensorflow as tf
from tensorflow.keras.losses import Loss
from tensorflow.keras.losses import BinaryCrossentropy

from sklearn.preprocessing import MinMaxScaler

class UnnormalizedBinaryCrossentropy(Loss):
    def __init__(self, **kwargs):
        super().__init__()
        kwargs['from_logits'] = True
        self._loss = BinaryCrossentropy(**kwargs)

    def call(self, y_true, y_pred):
        return self._loss.call(tf.math.sigmoid(y_true), y_pred)

    def get_config(self):
        config = self._loss.get_config()
        return config

if __name__ == "__main__":

    @scriptify
    def script(data_name,
               X_train=None,
               X_test=None,
               X_train_path=None,
               X_test_path=None,
               normalize=True,
               model_type='ae',
               arch='dense,d1024.d128',
               latent_dim=32,
               opt='adam',
               learning_rate=1.e-4,
               batch_size=512,
               epochs=100,
               loss='binary_crossentropy',
               gpu=0,
               save_to="weights/"):

        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
        device = gpus[gpu]

        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        if X_train_path is not None and X_train is None:
            X_train = np.load(X_train_path)

        if X_test_path is not None and X_test is None:
            X_test = np.load(X_test_path)

        if X_train is None:
            raise ValueError("X_train or the path to X_train must be provided.")

        if X_test is None:
            print("X_test is not provided, use training set as X_test")
            X_test = X_train

        data_range = [X_train.min(), X_train.max()]

        if normalize:
            scaler = MinMaxScaler(feature_range=(0, 1))
            scaler.fit(X_train)
            X_train = scaler.transform(X_train)
            X_test = scaler.transform(X_test)
            data_range = [X_train.min(), X_train.max()]
            vae_data_range =  [X_train.min(), X_train.max()]
        else:
            vae_data_range = data_range

        arch, arch_string = arch.split(',')[0], arch.split(',')[1]

        if arch == 'dense':
            X_train = np.reshape(X_train, (X_train.shape[0], -1))
            X_test = np.reshape(X_train, (X_train.shape[0], -1))

            input_shape = (X_train.shape[1], )
        else:
            if len(X_train.shape) < 4:
                X_train = X_train[:, :, :, None]
                X_test = X_test[:, :, :, None]
            input_shape = X_train.shape[1:]

        optimizer = tf.keras.optimizers.get(opt).__class__(
            learning_rate=learning_rate)

        if loss == 'unnormalized_bce':
            loss_fn = UnnormalizedBinaryCrossentropy()
        else:
            loss_fn = loss

        model = AE(input_shape,
                    latent_dim,
                    arch_string=arch_string,
                    normalize=vae_data_range)

        model.compile(optimizer=optimizer,
                      loss=loss_fn,
                      metrics=[tf.keras.losses.MeanSquaredError()])
        model.fit(X_train,
                  X_train,
                  batch_size=batch_size,
                  validation_data=(X_test, X_test),
                  epochs=epochs)

        model.save_weights(save_to + model_type + "_" + arch + "_" + data_name +
                           "_" + loss + ".h5")

        return {
            'final_weights':
            save_to + model_type + "_" + arch + "_" + data_name + "_" + loss +
            ".h5",
            'data_min':
            float(data_range[0]),
            'data_max':
            float(data_range[1])
        }