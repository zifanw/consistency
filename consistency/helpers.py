import numpy as np
import tensorflow as tf

class ProbitLayer(tf.keras.layers.Layer):
    def __init__(self):
        super(ProbitLayer, self).__init__()

    def build(self, input_shape):
        pass

    def call(self, inputs):
        return tf.concat([1.0 - inputs, inputs], axis=1)

def save_into_numpy(root, strategy, data, split, adv_x, y_pred_adv, is_adv):
    np.save(f"{root}/{strategy}_{data}_{split}_counterfactuals.npy", adv_x)
    np.save(f"{root}/{strategy}_{data}_{split}_counterfactuals_pred.npy",
            y_pred_adv)
    np.save(f"{root}/{strategy}_{data}_{split}_counterfactuals_is_adv.npy",
            is_adv)

def load_from_numpy(root, strategy, search_method, data, split):
    c = np.load(f"{root}/{strategy}_{data}_{search_method}_{split}.npz")
    return c['counterfactuals'], c['is_adv']


def unify_the_output_shape(model,
                           probit_output=False,
                           loss='categorical_crossentropy',
                           metrics=['acc'],
                           optimizer='adam'):

    re_compile = False
    inputs = model.input
    outputs = model.output

    if not probit_output:
        if len(outputs.shape) == 2 and outputs.shape[1] > 1:
            outputs = tf.keras.layers.Activation('softmax', name='softmax_pred')(outputs)
        else:
            outputs = tf.keras.activations.sigmoid(outputs)

        re_compile = True

    if outputs.shape[1] == 1:
        outputs = ProbitLayer()(outputs)
        re_compile = True

    if re_compile:
        re_model = tf.keras.models.Model(inputs, outputs)
        re_model.compile(loss=loss, metrics=metrics, optimizer=optimizer)
    else:
        re_model = model

    return re_model

def get_data_range(data):
    data_min_elementwise = data.min(axis=0)
    data_max_elementwise = data.max(axis=0)

    clamp = [data_min_elementwise, data_max_elementwise]
    return clamp 
