'''Setup: post-training evaluation of performance.'''

## External modules.
import numpy as np

## Internal modules.
from setup_losses import get_loss


###############################################################################


## Evaluation of the true (excess) risk in special case of quadratic linreg.

def risk_quadratic_linreg(w, w_star, A, b):
    risk_w = np.matmul(np.transpose(w-w_star),
                       np.matmul(A, (w-w_star)))/2 + b**2
    risk_w_star = b**2
    return risk_w-risk_w_star


## Evaluation metric parser.

def get_eval(loss_name=None, model_name=None, **kwargs):
    '''
    The default behaviour for this function is
    very simple; it just does evaluation using the
    loss that is used for training, taking the mean
    over all data points.
    In the special case of linear regression under
    the quadratic loss, then we also compute the
    exact risk (expected loss).
    '''
    eval_dict = {}
    if loss_name is not None:
        loss = get_loss(name=loss_name)
        loss_eval = lambda model, X, y: np.mean(loss(model=model,
                                                     X=X, y=y))
        eval_dict.update({loss_name: loss_eval})

        if loss_name == "quadratic" and model_name == "linreg":
            risk_eval = lambda model, X, y: risk_quadratic_linreg(
                w=model.paras["w"],
                w_star=kwargs["w_star"],
                A=kwargs["cov_X"],
                b=np.sqrt(kwargs["noise_var"])
            )
            eval_dict.update({"risk": risk_eval})
    
    return eval_dict


## Evaluation procedures.

def eval_model(epoch, model, model_idx, storage, data, eval_dict):

    ## Clean alias.
    j = model_idx
    
    ## Unpack things.
    X_train, y_train, X_test, y_test = data
    store_train, store_test = storage

    ## Carry out relevant evaluations.
    for key in store_train.keys():
        evaluator = eval_dict[key]
        store_train[key][epoch,j] = evaluator(model=model,
                                              X=X_train,
                                              y=y_train)
    for key in store_test.keys():
        evaluator = eval_dict[key]
        store_test[key][epoch,j] = evaluator(model=model,
                                             X=X_test,
                                             y=y_test)
    return None


def eval_models(epoch, models, storage, data, eval_dict):
    '''
    Loops over the model list, assuming enumerated index
    matches the performance array index.
    '''
    for j, model in enumerate(models):
        eval_model(epoch=epoch, model=model, model_idx=j,
                   storage=storage, data=data,
                   eval_dict=eval_dict)
    return None


## Sub-routine for writing to disk.

def eval_write(fname, storage, rb):

    ## Unpack.
    store_train, store_test = storage

    ## A string to distinguish which results are for the
    ## robustly boosted outputs.
    rb_str = "_rb" if rb else ""

    ## Write to disk as desired.
    if len(store_train) > 0:
        for key in store_train.keys():
            np.savetxt(fname=".".join([fname, key+"_train"+rb_str]),
                       X=store_train[key],
                       fmt="%.7e", delimiter=",")
    if len(store_test) > 0:
        for key in store_test.keys():
            np.savetxt(fname=".".join([fname, key+"_test"+rb_str]),
                       X=store_test[key],
                       fmt="%.7e", delimiter=",")
    return None


###############################################################################
