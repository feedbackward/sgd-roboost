'''Setup: preparation of simulated data sets.'''

## External modules.
import numpy as np

## Internal modules.
from mml.models.linreg import LinearRegression
from mml.utils.rgen import get_generator, get_stats


###############################################################################


## Detailed specification of data to be randomly generated.

_n = 500
_d = 2
_init_range = 5.0

dataset_paras = {
    "ds_lognormal": {
        "n_train": _n//2,
        "n_val": _n//2,
        "n_test": _n,
        "num_features": _d,
        "noise_dist": "lognormal",
        "noise_paras": {"mean": 0.0, "sigma": 1.75},
        "cov_X": np.eye(_d),
        "init_range": _init_range
    },
    "ds_normal": {
        "n_train": _n//2,
        "n_val": _n//2,
        "n_test": _n,
        "num_features": _d,
        "noise_dist": "normal",
        "noise_paras": {"loc": 0.0, "scale": 2.2},
        "cov_X": np.eye(_d),
        "init_range": _init_range
    }
}


## Data generation procedures.

def get_data(dataset, rg=None):
    '''
    Takes a string, return a tuple of data and parameters.
    '''
    try:
        paras = dataset_paras[dataset]
    except KeyError:
        raise ValueError(
            "Did not recognize dataset {}.".format(dataset)
        )

    return get_data_simulated(paras=paras, rg=rg)


def get_data_simulated(paras, rg=None):
    '''
    Data generation function.
    This particular implementation is a simple noisy
    linear model.
    '''

    n_train = paras["n_train"]
    n_val = paras["n_val"]
    n_test = paras["n_test"]
    n_total = n_train+n_val+n_test
    d = paras["num_features"]
    
    ## Setup of random generator.
    if rg is None:
        ss = np.random.SeedSequence()
        rg = np.random.default_rng(seed=ss)
    
    ## Specifying the true underlying model.
    w_star = np.ones(d).reshape((d,1))
    true_model = LinearRegression(num_features=d,
                                  paras_init={"w": w_star})
    paras.update({"w_star": w_star})
    
    ## Noise generator and stats.
    noise_gen = get_generator(name=paras["noise_dist"],
                              rg=rg,
                              **paras["noise_paras"])
    noise_stats = get_stats(name=paras["noise_dist"],
                            rg=rg,
                            **paras["noise_paras"])
    paras.update({"noise_mean": noise_stats["mean"],
                  "noise_var": noise_stats["var"]})

    ## Data generation.
    X = rg.multivariate_normal(mean=np.zeros(d),
                               cov=paras["cov_X"],
                               size=n_total)
    noise = noise_gen(n=n_total).reshape((n_total,1))-noise_stats["mean"]
    y = true_model(X=X) + noise
    
    ## Split into appropriate sub-views and return.
    X_train = X[0:n_train,...]
    y_train = y[0:n_train,...]
    X_val = X[n_train:(n_train+n_val),...]
    y_val = y[n_train:(n_train+n_val),...]
    X_test = X[(n_train+n_val):,...]
    y_test = y[(n_train+n_val):,...]
    return (X_train, y_train, X_val, y_val, X_test, y_test, paras)
    

###############################################################################
