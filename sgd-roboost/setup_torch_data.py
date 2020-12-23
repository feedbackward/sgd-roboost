'''Torch application: Data preparation.'''

## External modules.
import numpy as np
import os
from pathlib import Path
from tables import open_file
import torch
from torch.utils.data import TensorDataset


###############################################################################


## Clerical preparation.

# This directory will need to be set manually.
#dir_data_toread = os.path.join(str(Path.home()), "DATADIR")
dir_data_toread = os.path.join(str(Path.home()),
                               "tmp", "test_torch_demo",
                               "data_master")

# Specific dataset parameters that are set manually.
_n_train_frac = 0.8
_n_val_frac = 0.1*0.8

dataset_paras = {
    "adult": {"type": "classification",
              "chance_level": 0.7522, # freq of the majority class.
              "n_train_frac": _n_train_frac,
              "n_val_frac": _n_val_frac},
    "australian": {"type": "classification",
                   "chance_level": 0.5551, # freq of the majority class.
                   "n_train_frac": _n_train_frac,
                   "n_val_frac": _n_val_frac},
    "cifar10": {"type": "classification",
                "chance_level": 0.1,
                "pix_h": 32,
                "pix_w": 32,
                "channels": 3,
                "n_train_frac": _n_train_frac,
                "n_val_frac": _n_val_frac},
    "cod_rna": {"type": "classification",
                "chance_level": 0.6666, # freq of the majority class.
                "n_train_frac": _n_train_frac,
                "n_val_frac": _n_val_frac},
    "emnist_balanced": {"type": "classification",
                        "chance_level": 1/47,
                        "pix_h": 28,
                        "pix_w": 28,
                        "channels": 1,
                        "n_train_frac": _n_train_frac,
                        "n_val_frac": _n_val_frac},
    "ex_quad": {"type": "regression",
                "n_train_frac": 1.0},
    "fashion_mnist": {"type": "classification",
                      "chance_level": 0.1,
                      "pix_h": 28,
                      "pix_w": 28,
                      "channels": 1,
                      "n_train_frac": _n_train_frac,
                      "n_val_frac": _n_val_frac},
    "mnist": {"type": "classification",
              "chance_level": 0.1,
              "pix_h": 28,
              "pix_w": 28,
              "channels": 1,
              "n_train_frac": _n_train_frac,
              "n_val_frac": _n_val_frac}}


## Data-fetching function.

def get_data(dataset):
    '''
    Takes a string, return a tuple of data and parameters.
    '''
    try:
        paras = dataset_paras[dataset]
    except KeyError:
        raise ValueError(
            "Did not recognize dataset {}.".format(dataset)
        )

    if dataset == "ex_quad":
        return get_data_ex_quad(dataset=dataset, paras=paras)
    else:
        return get_data_general(dataset=dataset, paras=paras)
    

def get_data_general(dataset, paras):
    '''
    General purpose data-getter.
    '''
    
    ## Setup of random generator.
    ss = np.random.SeedSequence()
    rg = np.random.default_rng(seed=ss)
    
    toread = os.path.join(dir_data_toread, dataset,
                          "{}.h5".format(dataset))

    with open_file(toread, mode="r") as f:
        print(f)
        X = f.get_node(where="/", name="X").read().astype(np.float32)
        y = f.get_node(where="/", name="y").read().astype(np.int64).ravel()
        print("Types: X ({}), y ({}).".format(type(X), type(y)))
    
    ## If sample sizes are correct, then get an index for shuffling.
    if len(X) != len(y):
        s_err = "len(X) != len(y) ({} != {})".format(len(X),len(y))
        raise ValueError("Dataset sizes wrong. "+s_err)
    else:
        idx_shuffled = rg.permutation(len(X))
    
    X = X[idx_shuffled,:]
    y = y[idx_shuffled]
    
    ## Normalize the inputs in a per-feature manner (as max/min are vecs).
    maxvec = np.max(X, axis=0)
    minvec = np.min(X, axis=0)
    X = X-minvec
    with np.errstate(divide="ignore", invalid="ignore"):
        X = X / (maxvec-minvec)
        X[X == np.inf] = 0
        X = np.nan_to_num(X)
    del maxvec, minvec

    ## Get split sizes (training, validation, testing).
    n_all, num_features = X.shape
    print("(n_all, num_features) = {}".format((n_all, num_features)))
    n_train = int(n_all*paras["n_train_frac"])
    n_val = int(n_all*paras["n_val_frac"])
    n_test = n_all-n_train-n_val
    print("n_train = {}".format(n_train))
    print("n_val = {}".format(n_val))
    print("n_test = {}".format(n_test))

    ## Learning task specific parameter additions.
    paras.update({"num_features": num_features})
    if paras["type"] == "classification":
        paras.update({"num_classes": np.unique(y).size})
        print("num_classes = {}".format(paras["num_classes"]), "\n")

    ## Actually split the data (bench=training+validation, testing).
    X_bench = X[0:(n_train+n_val),:]
    X_test = X[(n_train+n_val):,:]
    y_bench = y[0:(n_train+n_val)]
    y_test = y[(n_train+n_val):]
    print("Types and shapes (before mapping):")
    print("X_bench: {} and {}.".format(type(X_bench), X_bench.shape))
    print("y_bench: {} and {}.".format(type(y_bench), y_bench.shape))
    print("X_test: {} and {}.".format(type(X_test), X_test.shape))
    print("y_test: {} and {}.".format(type(y_test), y_test.shape), "\n")
    
    ## Map from numpy arrays to torch tensors, and clear unneeded variables.
    X_bench, y_bench, X_test, y_test = map(
        torch.tensor,
        (X_bench, y_bench, X_test, y_test)
    )
    del X, y

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


def get_data_ex_quad(dataset, paras):
    '''
    A simple noise-free linear dataset for sanity checks.
    '''
    
    ## Setup of random generator.
    ss = np.random.SeedSequence()
    rg = np.random.default_rng(seed=ss)
    
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
