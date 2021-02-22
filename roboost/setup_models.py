'''Setup: models.'''

## External modules.
import numpy as np

## Internal modules.
from mml.models.linreg import LinearRegression


###############################################################################


## The main parser function, returning class objects (not instances).

_init_range = 5.0 # hard-coded here.

def get_model(name, paras_init=None, rg=None, **kwargs):

    ## Initializer preparation.
    if paras_init is None:
        try:
            ## If given w_star, use it (w/ noise).
            w_init = np.copy(kwargs["w_star"])
            w_init += rg.uniform(low=-_init_range,
                                 high=_init_range,
                                 size=w_init.shape)
            paras_init = {}
            paras_init["w"] = w_init
        except KeyError:
            ## If no w_star given, do nothing special.
            pass
    else:
        paras_init = paras_init

    ## Instantiate the desired model.
    if name == "linreg":
        return LinearRegression(num_features=kwargs["num_features"],
                                paras_init=paras_init, rg=rg)
    else:
        raise ValueError("Please pass a valid model name.")


###############################################################################
