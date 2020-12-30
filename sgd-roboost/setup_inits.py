'''Setup: initializers.'''

## External modules.
import numpy as np


###############################################################################


def get_w_init(rg, **kwargs):
    w_init = np.copy(kwargs["w_star"])
    noise = rg.uniform(low=-kwargs["init_range"],
                       high=kwargs["init_range"],
                       size=w_init.shape)
    return w_init + noise
                                             
    
###############################################################################
