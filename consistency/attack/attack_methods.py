import tensorflow as tf
import numpy as np
from tensorflow.python.types.core import Value

from attack.attack_utils import IGD_L1
from attack.attack_utils import IGD_L2
from attack.attack_utils import PGDs
from attack.attack_utils import SNS
from attack.attack_utils import search_z_adv


class Counterfactual(object):
    def __init__(self,
                 model,
                 clamp=[0, 1],
                 num_classes=2,
                 eps=0.3,
                 nb_iters=40,
                 eps_iter=0.01,
                 sns_fn=None):
        self.model = model
        self.clamp = clamp
        self.num_classes = num_classes
        self.eps = eps
        self.nb_iters = nb_iters
        self.eps_iter = eps_iter
        self.sns_fn = sns_fn

    def __call__(self, x, original_pred_sparse=None, batch_size=32, **kwargs):
        """
        :param x: (batch_size, num_features)
        :param original_pred_sparse: (batch_size, num_classes)
        :param batch_size:
        :param kwargs:
        :return:
        """

        self.y_sparse = self.get_original_prediction(x, original_pred_sparse)
        self.batch_size = batch_size

        # 1. generate counterfactual
        x_adv = self.generate_counterfactual(x,
                                             batch_size=batch_size,
                                             **kwargs)
        # Run SNS search
        if self.sns_fn is not None:
            x_adv = self.sns_fn(x_adv)

        # 2. check if counterfactual is adversarial
        pred_adv = np.argmax(self.model.predict(x_adv, batch_size=self.batch_size), -1)
        # 3. check if counterfactual is valid
        is_valid = self.is_valid(pred_adv)
        return x_adv, pred_adv, is_valid

    def get_original_prediction(self, x, original_pred_sparse):
        if original_pred_sparse is None:
            y_sparse = np.argmax(self.model.predict(x), -1)
        else:
            y_sparse = original_pred_sparse
        return y_sparse

    def generate_counterfactual(self, x, batch_size=32, **kwargs):
        raise NotImplementedError

    def is_valid(self, y):
        return self.y_sparse != y


class StableNeighborSearch(object):
    def __init__(self,
                 model,
                 clamp=[0, 1],
                 num_classes=2,
                 sns_eps=0.3,
                 sns_nb_iters=100,
                 sns_eps_iter=1.e-3,
                 n_interpolations=20):

        self.model = model
        self.clamp = clamp
        self.num_classes = num_classes
        self.sns_eps = sns_eps
        self.sns_nb_iters = sns_nb_iters
        self.sns_eps_iter = sns_eps_iter
        self.n_interpolations = n_interpolations

    def __call__(self, x):
        adv_x, _, _ = SNS(
            self.model,
            x,
            np.argmax(self.model.predict(x, batch_size=self.batch_size), -1),
            clamp=self.clamp,
            num_class=self.num_classes,
            batch_size=self.batch_size,
            n_steps=self.n_interpolations,
            max_steps=self.sns_nb_iter,
            adv_epsilon=self.sns_eps,
            adv_step_size=self.sns_eps_iter)
        return adv_x


class IterativeSearch(Counterfactual):
    """Summary goes here"""
    def __init__(self, *args, norm=2, **kwargs):
        super().__init__(*args, **kwargs)
        self.norm = norm

    def generate_counterfactual(self, x, batch_size, **kwargs):
        if self.norm == 1:
            adv_x, _, is_adv = IGD_L1(self.model,
                                      x,
                                      self.y_sparse,
                                      num_classes=self.num_classes,
                                      steps=self.nb_iters,
                                      clamp=self.clamp,
                                      batch_size=batch_size,
                                      **kwargs)
        elif self.norm == 2:
            adv_x, _, is_adv = IGD_L2(self.model,
                                      x,
                                      self.y_sparse,
                                      num_classes=self.num_classes,
                                      steps=self.nb_iters,
                                      clamp=self.clamp,
                                      batch_size=batch_size,
                                      **kwargs)
        else:
            raise ValueError("norm must be integers (1 or 2)")

        adv_x = adv_x[is_adv]

        return adv_x


class PGDsL2(Counterfactual):
    """Summary goes here"""
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def generate_counterfactual(self, x, batch_size, **kwargs):
        adv_x, _, is_adv = PGDs(self.model,
                                x,
                                self.y_sparse,
                                epsilon=self.eps,
                                stepsize=self.eps_iter,
                                num_classes=self.num_classes,
                                steps=self.nb_iters,
                                clamp=self.clamp,
                                batch_size=batch_size,
                                **kwargs)

        if np.sum(is_adv) < 1:
            raise RuntimeError("No adversarial samples found")
        else:
            adv_x = adv_x[is_adv]
            return adv_x


class LatentSpaceSearch(Counterfactual):
    """Summary goes here"""
    def __init__(self, encoder_encoder, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.encoder_encoder = encoder_encoder

    def generate_counterfactual(self, x, batch_size, **kwargs):
        z = self.encoder_decoder.encode(x)
        adv_x, _, _ = search_z_adv(self.encoder_decoder,
                                   self.model,
                                   x,
                                   z,
                                   self.y_sparse,
                                   clamp=self.clamp,
                                   num_class=self.num_classes,
                                   batch_size=batch_size,
                                   **kwargs)

        return adv_x


class Prototype(Counterfactual):
    """Summary goes here"""
    def __init__(self, cf, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.cf = cf

    def generate_counterfactual(self, x, batch_size, **kwargs):

        del batch_size

        adv_x = []
        c_labels = []

        pb = tf.keras.utils.Progbar(target=x.shape[0])
        for x_in in x:
            explanation = self.cf.explain(x_in[None, :], **kwargs)
            adv_x.append(explanation['cf']['X'])
            c_labels.append(explanation['cf']['class'])
            pb.add(1)
        adv_x = np.vstack(adv_x)

        return adv_x
