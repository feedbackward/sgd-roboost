'''Performance-related helper functions.'''

## External modules.
from numpy import savetxt
import torch
import torch.nn.functional as F


###############################################################################


## Additional performance-related functions.

def accuracy(scores, labels):
    return (torch.argmax(scores, dim=1) == labels).float().mean()


## Dictionaries organizing the functions used for evaluation.

dict_losses = {"nll": F.nll_loss,
               "mse": F.mse_loss,
               "cross_entropy": F.cross_entropy}


## The main parser function.

def get_eval(loss_fn, task_type):
    out_loss_fn = dict_losses[loss_fn]
    eval_fn_dict = {}
    if task_type == "classification":
        eval_fn_dict.update({"acc": accuracy})
    return out_loss_fn, eval_fn_dict


## Evaluation procedures.

def eval_model(epoch, model, model_idx, loss_arrays, eval_dicts, data,
               device, loss_fn, eval_fn_dict):

    ## Clean alias.
    j = model_idx
    
    ## Unpack things.
    X_train, y_train, X_test, y_test = data
    losses_train, losses_test = loss_arrays
    eval_train_dict, eval_test_dict = eval_dicts
    
    ## Then get to evaluation.
    model.eval()
    with torch.no_grad():
        pred_train = model(X_train.to(device))
        target_train = y_train.to(device)
        losses_train[epoch,j] = float(
            loss_fn(pred_train, target_train).item()
        )
        if losses_test is not None:
            pred_test = model(X_test.to(device))
            target_test = y_test.to(device)
            losses_test[epoch,j] = float(
                loss_fn(pred_test, target_test).item()
            )
        if len(eval_train_dict) > 0:
            for key in eval_train_dict.keys():
                eval_fn = eval_fn_dict[key]
                eval_train_dict[key][epoch,j] = float(
                    eval_fn(pred_train, target_train).item()
                )
                if eval_test_dict is not None:
                    eval_test_dict[key][epoch,j] = float(
                        eval_fn(pred_test, target_test).item()
                    )
        return None


def eval_models(epoch, models, loss_arrays, eval_dicts, data,
                device, loss_fn, eval_fn_dict):

    ## Loops over the model list, assuming enumerated index
    ## matches the performance array index.
    for j, model in enumerate(models):
        eval_model(epoch=epoch, model=model, model_idx=j,
                   loss_arrays=loss_arrays, eval_dicts=eval_dicts,
                   data=data, device=device, loss_fn=loss_fn,
                   eval_fn_dict=eval_fn_dict)


## Sub-routine for writing to disk ("rb" is roboost-related).

def eval_write(fname, losses_train, losses_test,
               eval_train_dict, eval_test_dict, rb):

    rb_str = "_rb" if rb else ""
    
    savetxt(fname=".".join([fname, "losses_train"+rb_str]),
            X=losses_train,
            fmt="%.7e", delimiter=",")
    if losses_test is not None:
        savetxt(fname=".".join([fname, "losses_test"+rb_str]),
                X=losses_test,
                fmt="%.7e", delimiter=",")
    if len(eval_train_dict) > 0:
        for key in eval_train_dict.keys():
            savetxt(fname=".".join([fname, key+"_train"+rb_str]),
                    X=eval_train_dict[key],
                    fmt="%.7e", delimiter=",")
    if eval_test_dict is not None:
        for key in eval_test_dict.keys():
            savetxt(fname=".".join([fname, key+"_test"+rb_str]),
                    X=eval_test_dict[key],
                    fmt="%.7e", delimiter=",")


###############################################################################
