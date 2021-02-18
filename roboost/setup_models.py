'''Setup: models.'''

## External modules.
import numpy as np

## Internal modules.
from mml.models.linreg import LinearRegression


###############################################################################


## The main parser function, returning class objects (not instances).

def get_model(model_class, paras_init=None, rg=None, **kwargs):

    if model_class == "linreg":
        return LinearRegression(num_features=kwargs["num_features"],
                                paras_init=paras_init, rg=rg)
    else:
        raise ValueError("Please pass a valid model name.")


###############################################################################
