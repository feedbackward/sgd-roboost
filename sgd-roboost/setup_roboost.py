'''Setup: robust boosting sub-routines.'''

## External modules.
import numpy as np

## Internal modules.
from mml.utils.mest import est_loc_fixedpt, inf_catwide, est_scale_chi_fixedpt, chi_geman_quad
from mml.utils.vecmean import geomed, geomed_set, smallball


###############################################################################


## List of names of roboost methods to be tried.
todo_roboost = ["take-rand",
                "geomed-space", "geomed-set", "smallball", "centroid",
                "valid-ave", "valid-med", "valid-robust"]

## Note: all possibilities are listed below.
#["triv-first", "triv-last", "take-rand",
# "geomed-space", "geomed-set", "smallball", "centroid",
# "valid-ave", "valid-med", "valid-robust"]


## Main routine definition.

def do_roboost(model_todo, ref_models, cand_array,
               data_val, loss, rb_method, rg):

    ## Before doing anything, check if output is trivial.
    if cand_array.shape[0] == 1:
        model_todo.w = cand_array[0:1,:].T
        return None

    ## Otherwise, proceed.
    is_valid = True if rb_method.split("-")[0] == "valid" else False
    
    if is_valid:
        X_val, y_val = data_val
        est_type = rb_method.split("-")[1]
        losses = []
        for model in ref_models:
            losses.append(loss(model=model, X=X_val, y=y_val))
        losses = np.column_stack(losses) # shape is (n_val, k).
        if est_type == "ave":
            loss_stats = np.mean(losses, axis=0)
        elif est_type == "med":
            loss_stats = np.median(losses, axis=0)
        elif est_type == "robust":
            s = est_scale_chi_fixedpt(
                X=losses-np.mean(losses, axis=0),
                chi_fn=chi_geman_quad
            )
            loss_stats = est_loc_fixedpt(X=losses, s=s,
                                         inf_fn=inf_catwide).ravel()
        else:
            raise ValueError("Please pass a proper validation subroutine.")
        p_new = cand_array[np.argmin(loss_stats),:]
            
    elif rb_method == "geomed-space":
        p_new = geomed(A=cand_array).ravel()
    elif rb_method == "geomed-set":
        p_new = geomed_set(A=cand_array).ravel()
    elif rb_method == "smallball":
        p_new = smallball(A=cand_array).ravel()
    elif rb_method == "centroid":
        p_new = np.mean(cand_array, axis=0)
    elif rb_method == "take-rand":
        idx_rand = rg.choice(a=len(cand_array), size=1).item()
        p_new = np.copy(cand_array[idx_rand,:])
    elif rb_method == "triv-first":
        p_new = np.copy(cand_array[0,:]) ## trivial, for debugging.
    elif rb_method == "triv-last":
        p_new = np.copy(cand_array[-1,:]) ## trivial, for debugging.
    else:
        raise ValueError(
            "Please pass a valid rb_method; got {}. ({})".format(
                rb_method, todo_roboost
            )
        )
    
    model_todo.w = p_new.reshape(model_todo.w.shape)
    return None


###############################################################################
