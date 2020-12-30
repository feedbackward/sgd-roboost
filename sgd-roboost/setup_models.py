'''Setup: models.'''

## External modules.
import numpy as np

## Internal modules.
from mml.models.linreg import LinearRegression


###############################################################################


## Dictionary for organizing model class objects.

models_dict = {"linreg": LinearRegression}


## The main parser function, returning class objects (not instances).

def get_model(model_class):
    return models_dict[model_class]


###############################################################################
