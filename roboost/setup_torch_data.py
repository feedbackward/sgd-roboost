'''Torch application: Data preparation.'''

## External modules.
import numpy as np
import os
from pathlib import Path
from tables import open_file
import torch
from torch.utils.data import TensorDataset

## Internal modules.
from mml.data import dataset_dict, dataset_list, get_data_general


###############################################################################


## If benchmark data is to be used, specify the directory here.
dir_data_toread = os.path.join(str(Path.home()), "DATADIR")


## First set dataset parameter dictionary with standard values
## for all the benchmark datasets in mml.
dataset_paras = dataset_dict


## Then add customized parameters for the simulations local to this project.
dataset_paras_local = {
    "ex_quad": {"type": "regression",
                "n_train_frac": 1.0}
}
dataset_paras.update(dataset_paras_local)


## Data generation procedure.

def get_data(dataset, rg):
    '''
    Takes a string, return a tuple of data and parameters.
    '''
    if dataset in dataset_paras:
        paras = dataset_paras[dataset]
        if dataset in dataset_list:
            ## Benchmark dataset case.
            return get_data_general_torch(dataset=dataset,
                                          paras=paras, rg=rg,
                                          directory=dir_data_toread)
        else:
            ## Local simulation case.
            return get_data_ex_quad(paras=paras, rg=rg)
    else:
        raise ValueError(
            "Did not recognize dataset {}.".format(dataset)
        )
        

def get_data_general_torch(dataset, paras, rg, directory):
    '''
    A local torch wrapper for our general purpose data-getter.
    '''

    X_train, y_train, X_val, y_val, X_test, y_test, paras = get_data_general(
        dataset=dataset, paras=paras, rg=rg, directory=directory,
        do_normalize=True, do_shuffle=True, do_onehot=False
    )
    if y_train.shape[1] > 1:
        raise ValueError("Assumes only one label for each data point.")
    
    n_train = len(X_train)
    n_val = len(X_val)
    
    X_bench = np.vstack((X_train,X_val))
    y_bench = np.vstack((y_train,y_val))

    ## Map from numpy arrays to torch tensors, and clear unneeded variables.
    ## Also flatten as needed.
    X_bench, y_bench, X_test, y_test = map(
        torch.tensor,
        (X_bench, y_bench.reshape(-1), X_test, y_test.reshape(-1))
    )
    del X_train, y_train, X_val, y_val
    
    ## Get views of relevant training and validation subsets.
    X_train = X_bench[0:n_train,:]
    X_val = X_bench[n_train:,:]
    y_train = y_bench[0:n_train]
    y_val = y_bench[n_train:]
    print("Types and shapes (after mapping):")
    print("X_bench: {} and {}.".format(type(X_bench), X_bench.shape))
    print("y_bench: {} and {}.".format(type(y_bench), y_bench.shape))
    print("X_test: {} and {}.".format(type(X_test), X_test.shape))
    print("y_test: {} and {}.".format(type(y_test), y_test.shape), "\n")

    return (X_bench, y_bench, X_train, y_train, X_val, y_val,
            X_test, y_test, paras)


def get_data_ex_quad(paras, rg):
    '''
    A simple noise-free linear dataset for sanity checks.
    '''
    
    n_tr = 45
    n_val = 15
    n_te = 15
    
    ## First, generate some data.
    w_star = np.array([1., 2., 3.], dtype=np.float32).reshape((3,1))
    num_features = len(w_star)
    X_bench = rg.standard_normal((n_tr+n_val,num_features),
                                 dtype=np.float32)
    X_test = rg.standard_normal((n_te,num_features), dtype=np.float32)
    y_bench = np.matmul(X_bench,w_star)
    y_test = np.matmul(X_test,w_star)
    
    ## Map from numpy arrays to torch tensors.
    X_bench, y_bench, X_test, y_test = map(
        torch.tensor,
        (X_bench, y_bench, X_test, y_test)
    )
    X_train = X_bench[0:n_tr,:]
    X_val = X_bench[n_tr:,:]
    y_train = y_bench[0:n_tr]
    y_val = y_bench[n_tr:]
    paras.update({"num_features": num_features,
                  "num_classes": 1})
    return (X_bench, y_bench, X_train, y_train,
            X_val, y_val, X_test, y_test, paras)


###############################################################################
