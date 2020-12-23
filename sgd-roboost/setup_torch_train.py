'''Torch application: training procedures.'''

## External modules.
import numpy as np
import os
from tables import open_file


###############################################################################


def train_epoch(num, model, algo, data_loader, loss_fn, device, verbose=False):
    '''
    Basic procedure for a single epoch (one run through data loader).
    '''

    if verbose:
        print(
            "Training... (num={},\tdevice={}). Model: {}".format(
                num, device, model
            )
        )
    else:
        print(
            "Training... (num={},\tdevice={}).".format(num, device)
        )

    model.train()
    pid = os.getpid()
    for batch_num, (xb, yb) in enumerate(data_loader):
        algo.zero_grad()
        loss = loss_fn(model(xb.to(device)), yb.to(device))
        loss.backward()
        algo.step()
        algo.zero_grad()

    return None


###############################################################################
