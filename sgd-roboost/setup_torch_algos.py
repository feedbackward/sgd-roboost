'''Algorithm management for PyTorch application.'''

## External modules.
import torch.optim as optim


###############################################################################


def get_algo(algo_name, model, **kwargs):
    
    if algo_name == "SGD":
        return optim.SGD(model.parameters(),
                         lr=kwargs["step_size"],
                         momentum=kwargs["momentum"])
    else:
        return None


###############################################################################
