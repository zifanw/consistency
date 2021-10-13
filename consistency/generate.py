import os
from os import path

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import tensorflow as tf
import numpy as np
from sklearn.preprocessing import MinMaxScaler

from helpers import unify_the_output_shape
from helpers import load_from_numpy
from helpers import save_into_numpy
from helpers import get_data_range

from attack import search_counterfactuals
import latent_space.models as GM
from metrics import invalidation
from scriptify import scriptify


def generate_counterfactuals(model, x, y_sparse, **kwargs):

    scaler = MinMaxScaler()
    scaler.fit(x)

    if kwargs['vae'] is not None:
        x = scaler.transform(x)

    c, _, is_adv = search_counterfactuals(model,
                                          x,
                                          y_sparse,
                                          vae_samples=256,
                                          return_probits=True,
                                          transform=scaler.inverse_transform,
                                          **kwargs)
    return c, is_adv

if __name__ == "__main__":

    @scriptify
    def script(data_x,
               data_y,
               path_base_model,
               path_comparing_models,
               vae_hp_dict,
               grad_hp_dict,
               sns_hp_dict,
               probit_output=True,
               model_restore_fn=tf.keras.models.load_model,
               use_pred_as_labels=True,
               overwrite=False,
               batch_size=256,
               gpu=0,
               search_method="CW",
               save_path="results"):

        if search_method == 'Proto':
            tf.compat.v1.disable_eager_execution()

        gpus = tf.config.experimental.list_physical_devices('GPU')
        tf.config.experimental.set_visible_devices(gpus[gpu], 'GPU')
        device = gpus[gpu]

        for device in tf.config.experimental.get_visible_devices('GPU'):
            tf.config.experimental.set_memory_growth(device, True)

        clamp = get_data_range(data_x)

        modelA = model_restore_fn(path_base_model)
        modelA = unify_the_output_shape(modelA, probit_output)

        if use_pred_as_labels:
            data_y_sparse = np.argmax(
                modelA.predict(data_x, batch_size=batch_size), -1)

        else:
            modelA_counterfactuals, is_adv = generate_counterfactuals(
                vae_hp_dict,
                grad_hp_dict,
                sns_hp_dict,
                modelA,
                data_x.copy(),
                data_y_sparse.copy(),
                )

        modelA_counterfual_probits = modelA.predict(modelA_counterfactuals,
                                                    batch_size=batch_size)
        modelA_counterfual_preds = np.argmax(modelA_counterfual_probits, -1)

        np.savez(
            f"{save_path}/{baseline_strategy}_{data}_{search_method}_{split}.npz",
            counterfactuals=modelA_counterfactuals,
            is_adv=np.array(is_adv),
            pred=modelA_counterfual_preds,
            confidence=np.max(modelA_counterfual_probits, -1))

        data_y_sparse = data_y_sparse[is_adv]
        data_x = data_x[is_adv]

        ################################################################
        # iv: invalidation rate
        # mBc_c: counfidence of counterfactuals predicted by model B
        # d_x2c: distance bewteen data and counterfactuals
        ################################################################

        all_iv = []
        all_mBc_c = []

        for t in comparing_strategy.split(','):
            if t == 'baseline':
                modelB = tf.keras.models.load_model(model_paths.baseline)
                modelB = unify_the_output_shape(modelB, model_paths)

                iv, (mBc_c, mBc_p) = invalidation(modelA_counterfactuals,
                                                  modelA_counterfual_preds,
                                                  modelB,
                                                  batch_size=batch_size,
                                                  aggregation=None,
                                                  return_pred_B=True)

                all_iv.append(iv)
                all_mBc_c.append(mBc_c.mean())

                np.savez(f"{save_path}/{t}_{data}_{search_method}_{split}.npz",
                         counterfactuals=modelA_counterfactuals,
                         pred=mBc_p,
                         confidence=mBc_c)

            elif t == 'loo':
                for i in range(number_of_models):
                    modelB = tf.keras.models.load_model(
                        model_paths.loo_list[i])
                    modelB = unify_the_output_shape(modelB, model_paths)

                    iv, (mBc_c, mBc_p) = invalidation(modelA_counterfactuals,
                                                      modelA_counterfual_preds,
                                                      modelB,
                                                      batch_size=batch_size,
                                                      aggregation=None,
                                                      return_pred_B=True)

                    all_iv.append(iv)
                    all_mBc_c.append(mBc_c.mean())

                    np.savez(f"{save_path}/{t}_{i}_{data}_{split}.npz",
                             counterfactuals=modelA_counterfactuals,
                             pred=mBc_p,
                             confidence=mBc_c)

            elif t == 'rs':
                for i in range(number_of_models):
                    modelB = tf.keras.models.load_model(model_paths.rs_list[i])
                    modelB = unify_the_output_shape(modelB, model_paths)

                    iv, (mBc_c, mBc_p) = invalidation(modelA_counterfactuals,
                                                      modelA_counterfual_preds,
                                                      modelB,
                                                      batch_size=batch_size,
                                                      aggregation=None,
                                                      return_pred_B=True)

                    all_iv.append(iv)
                    all_mBc_c.append(mBc_c.mean())

                    np.savez(
                        f"{save_path}/{t}_{i}_{data}_{search_method}_{split}.npz",
                        counterfactuals=modelA_counterfactuals,
                        pred=mBc_p,
                        confidence=mBc_c)

            else:
                raise ValueError(f"{t} is not a valid training_strategy")

        avg_iv, avg_std = np.mean(all_iv), np.std(all_iv)

        avg_mBc_c = np.mean(all_mBc_c)
        avg_d_x2c = np.mean(
            np.linalg.norm(data_x - modelA_counterfactuals, axis=-1))
        avg_d_x2c_std = np.std(
            np.linalg.norm(data_x - modelA_counterfactuals, axis=-1))
        avg_d_x2c_l1 = np.mean(
            np.linalg.norm(data_x - modelA_counterfactuals, axis=-1, ord=1))
        avg_d_x2c_l1_std = np.std(
            np.linalg.norm(data_x - modelA_counterfactuals, axis=-1, ord=1))
        avg_mAc_c = np.max(modelA_counterfual_probits, -1).mean()
        success_rate = np.mean(is_adv)

        print("\n\n")
        print(
            ">>>>>>>>>>>>>>>>>>>>>>>>>>>>>>>> Result <<<<<<<<<<<<<<<<<<<<<<<<<<<<<<<"
        )
        print(
            f"Success Rate [EPS={epsilon}, {search_method}]:     {success_rate}"
        )
        print(f"Invalidation Rate:     {avg_iv}+-{avg_std}")
        print(
            f"Confidence on Counterfactual ({baseline_strategy}):     {avg_mAc_c}"
        )
        print(
            f"Confidence on Counterfactual ({comparing_strategy}):     {avg_mBc_c}"
        )
        print(
            f"L2 Distance (Data Range: {np.min(clamp[0]), np.max(clamp[1])}):     {avg_d_x2c}+-{avg_d_x2c_std}"
        )
        print(
            f"L1 Distance (Data Range: {np.min(clamp[0]), np.max(clamp[1])}):     {avg_d_x2c_l1}+-{avg_d_x2c_l1_std}"
        )

        return_dict = {
            "IV": round(float(avg_iv), 2),
            "IV_std": round(float(avg_std), 2),
            "mAc_c": round(float(avg_mAc_c), 2),
            "mBc_c": round(float(avg_mBc_c), 2),
            "d_x2c": round(float(avg_d_x2c), 4),
            "d_x2c_l1": round(float(avg_d_x2c_l1), 4),
            "d_x2c_std": round(float(avg_d_x2c_std), 4),
            "d_x2c_l1_std": round(float(avg_d_x2c_l1_std), 4),
            "success_rate": round(float(success_rate), 2),
        }

        return return_dict
