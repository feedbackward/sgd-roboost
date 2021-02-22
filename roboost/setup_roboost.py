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

## Main routine definition.

def do_roboost(model_todo, ref_models, cand_dict,
               data_val, loss, rb_method, rg):

    for pn, cand_array in cand_dict.items():

        ## Record the parameter shape.
        out_shape = model_todo.paras[pn].shape

        ## Before doing anything, check if output is trivial.
        if len(cand_array) == 1:
            p_new = cand_array[0,...]
            model_todo.paras[pn] = np.copy(p_new)
            continue
    
        ## Otherwise, proceed.
        do_validation = True if "valid" in rb_method else False
    
        if do_validation:
            ## If doing validation, use data set aside for this.
            X_val, y_val = data_val
            est_type = rb_method.split("-")[1]
            losses = []
            for model in ref_models:
                losses.append(loss(model=model, X=X_val, y=y_val))
            losses = np.column_stack(losses) # shape is (n_val, num_processes).
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
            p_new = cand_array[np.argmin(loss_stats),...]
            
        elif rb_method == "geomed-space":
            p_new = geomed(A=cand_array)
        elif rb_method == "geomed-set":
            p_new = geomed_set(A=cand_array)
        elif rb_method == "smallball":
            p_new = smallball(A=cand_array)
        elif rb_method == "centroid":
            p_new = np.mean(cand_array, axis=0, keepdims=False)
        elif rb_method == "take-rand":
            idx_rand = rg.choice(a=len(cand_array), size=1).item()
            p_new = cand_array[idx_rand,...]
        elif rb_method == "triv-first":
            p_new = cand_array[0,...] ## trivial, for debugging.
        elif rb_method == "triv-last":
            p_new = cand_array[-1,...] ## trivial, for debugging.
        else:
            raise ValueError(
                "Please pass a valid rb_method; got {}. ({})".format(
                    rb_method, todo_roboost
                )
            )
        
        ## Final shape check before updating with a copy.
        if p_new.shape != out_shape:
            raise RuntimeError("p_new.shape {}".format(p_new.shape))
        else:
            model_todo.paras[pn] = np.copy(p_new)
    
    return None


###############################################################################
