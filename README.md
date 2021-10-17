# Consistency

Implementation of [*Consistent Counterfactuals for Deep Model*](https://arxiv.org/pdf/2110.03109.pdf) with TensorFlow 2.

![Test Image 1](media/results.png)

## Setup

### Tensorflow Version 

All code is tested under TF version `2.3.0`.

### Install the Dependency
The following command will install `foolbox` and `data;ib`.
```
sh setup.sh
```

### Prepare the dataset

The following dataset can be downloaded by running the command.

```
sh get_data.sh
```

- German Credit (binary classification)
- Seizure (binary classification)
- CTG (binary classification)
- HELOC (binary classification)
- Warfarin (three classes)
- Taiwanese Credit (binary classification)

## Stable Neighbor Search
Stable Neighbor Search can be easily plugged into the iterative adversarial methods provided in this repository. The following code demonstrates an example. 

```python3
from consistency import PGDsL2
from consistency import StableNeighborSearch
from utils import load_dataset

# Load the tf.keras model
model = load_your_model(...)

# Load dataset  

(X_train, y_train), (X_test, y_test), n_classes = load_dataset('Seizure', path_to_data_dir='dataset/data')

# Configue the stable neighbor search parameters.
sns_fn = StableNeighborSearch(model,
                 clamp=[X_train.min(), X_train.max()], # valid data range
                 num_classes=2,
                 sns_eps=0.1,         # the epsilon bound
                 sns_nb_iters=100,    # maximum number of steps
                 sns_eps_iter=1.e-3,  # step size
                 n_interpolations=20) # number of samples to approximate the integral

# Run the PGD attack
pgd_iter_search = PGDsL2(model,
                        clamp=[X_train.min(), X_train.max()],
                        num_classes=2,
                        eps=2.0,
                        nb_iters=100,
                        eps_iter=0.04,
                        sns_fn=sns_fn)
pgd_cf, pred_cf, is_valid = pgd_iter_search(X_test[:128], num_interpolations=10, batch_size=64)
```

For the full example, check out `example.ipynb`. 

For the detail of hyper-parameters, check out the paper. The hyper-parameters used in the code block above are only for demonstration purpose.

## Evaluation 

To evaluate the invalidation rate, one can use the following code.

```python3
from utils import invalidation
model_1 = load_your_other_model(...)

iv = invalidation(pgd_cf,
                np.argmax(baseline_model.predict(X_test[:128]), axis=1),
                model_1,
                batch_size=32,
                affinity_set=[[0], [1]])  # Define what classes are considered as counerfactual to the others.

```

