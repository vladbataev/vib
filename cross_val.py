import numpy as np
from vib import run_experiment


def main():
    params_grid = []
    betas = np.logspace(-9, 0, 10)
    for beta in betas:
        for use_stoch in [[True, True, True], [False, False, True], [False, True, True]]:
            params = {
                "stoch_z_dims": [512, 256, 128],
                "num_epochs": 300,
                "use_stoch": use_stoch,
                "betas": [beta] * 3,
                "num_dense_layers": 3,
                "early_stopping": False,
                "betas_equal": False,
            }
            params_grid.append(params)
    for params in params_grid:
        run_experiment(params)


if __name__ == "__main__":
    main()
