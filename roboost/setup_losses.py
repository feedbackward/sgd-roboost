'''Setup: loss functions used for training.'''

## Internal modules.
from mml.losses.quadratic import Quadratic


###############################################################################


## A dictionary of instantiated losses.

dict_losses = {"quadratic": Quadratic()}

def get_loss(name):
    '''
    A simple parser that returns a loss instance.
    '''
    return dict_losses[name]


###############################################################################
