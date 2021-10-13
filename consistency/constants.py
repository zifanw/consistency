HP_GRADIENT_ATTACK = {
    "epsilon": 1.0,
    "norm": 2,
    "num_interpolation": 10,
    "max_steps": 300,
    "k_tree": 20,
    "step_size": None,
    "return_probits": True
}

HP_AE_ATTACK = {
    "vae_arch_string": None,
    "vae_weights": None,
    "epsilon": 1.0,
    "norm": 2,
    "num_samples": 256,
    "max_steps": 5,
    "detemintristic": True,
    "direction": "random",
    "return_probits": True
}

HP_SNS = {
    "eps_ratio": 0.8,
    "step_size": None,
    "max_steps": 300,
    "norm": 2,
    "res": 10,
    "a": 0.5,
    "return_probits": True
}