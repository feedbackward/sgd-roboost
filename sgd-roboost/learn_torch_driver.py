'''Driver script for learning algorithms implemented using PyTorch.'''

## External modules.
import argparse
from copy import deepcopy
import json
import numpy as np
import os
import torch
import torch.multiprocessing as mp
from torch.utils.data import TensorDataset, DataLoader

## Internal modules.
from mml.utils import makedir_safe
from setup_results import results_dir
from setup_torch_algos import get_algo
from setup_torch_data import get_data
from setup_torch_eval import get_eval, eval_models, eval_model, eval_write
from setup_torch_models import get_model
from setup_torch_roboost import do_roboost, todo_roboost, rg_roboost
from setup_torch_train import train_epoch


###############################################################################


## Basic setup.

parser = argparse.ArgumentParser(description="Arguments for driver script.")

parser.add_argument("--algo",
                    help="Algorithm class. (default: SGD)",
                    type=str, default="SGD", metavar="S")
parser.add_argument("--batch-size",
                    help="Mini-batch size for iterative optimizers.",
                    type=int, default=8, metavar="N")
parser.add_argument("--cuda",
                    help="Enables CUDA-based training (default: False).",
                    action="store_true", default=False)
parser.add_argument("--data",
                    help="Specify data set to be used (default: None).",
                    type=str, default=None, metavar="S")
parser.add_argument("--loss-fn",
                    help="Loss function name. (default: nll)",
                    type=str, default="nll", metavar="S")
parser.add_argument("--model",
                    help="Model class. (default: logistic)",
                    type=str, default="logistic", metavar="S")
parser.add_argument("--momentum",
                    help="Momentum parameter (default: 0.0)",
                    type=float, default=0.0, metavar="F")
parser.add_argument("--num-epochs",
                    help="Number of epochs to run (default: 3)",
                    type=int, default=3, metavar="N")
parser.add_argument("--num-processes",
                    help="Number of learning sub-processes (default: 1)",
                    type=int, default=1, metavar="N")
parser.add_argument("--num-trials",
                    help="Number of independent random trials (default: 1)",
                    type=int, default=1, metavar="N")
parser.add_argument("--seed",
                    help="Seed for torch RNGs.",
                    type=int, default=42, metavar="N")
parser.add_argument("--step-size",
                    help="Step size parameter (default: 0.01)",
                    type=float, default=0.01, metavar="F")
parser.add_argument("--task-name",
                    help="A task name. Default is the word default.",
                    type=str, default="default", metavar="S")
parser.add_argument("--verbose",
                    help="Print details or not (default: False).",
                    action="store_true", default=False)
                    

## Parse the arguments passed via command line.
args = parser.parse_args()
if args.data is None:
    raise TypeError("Given --data=None, should be a string.")

## Name to be used identifying the results etc. of this experiment.
towrite_name = args.task_name+"-"+"_".join([args.model, args.algo])

## Model class must be initialized here, to ensure all sub-procs get access.
Model_class, paras_todo = get_model(model_class=args.model)

## Prepare a directory to save results.
towrite_dir = os.path.join(results_dir, "torch", args.data)
makedir_safe(towrite_dir)

## Main process.

if __name__ == "__main__":

    ## Device settings.
    use_cuda = args.cuda and torch.cuda.is_available()
    dev = torch.device("cuda" if use_cuda else "cpu")
    print("cuda.is_available():", torch.cuda.is_available())
    print("args.cuda:", args.cuda)
    print("use_cuda:", use_cuda, "\n")
    
    ## Arguments for the data loaders.
    dl_kwargs = {"batch_size": args.batch_size,
                 "shuffle": True}
    if use_cuda:
        dl_kwargs.update({"num_workers": 1,
                          "pin_memory": True})
    
    ## Arguments for algorithms.
    algo_kwargs = {"step_size": args.step_size,
                   "momentum": args.momentum}
    
    #torch.manual_seed(args.seed) ## for reproducing results; not yet used.
    ctx = mp.get_context("spawn") # following python docs.

    ## Start the loop over independent trials.

    for trial in range(args.num_trials):

        ## Load in data as (pre-shuffled) tensors.
        print("Start data prep.")
        print("args.data:", args.data)
        (X_bench, y_bench, X_train, y_train, X_val, y_val,
         X_test, y_test, ds_paras) = get_data(args.data)
        print("Is contiguous?")
        print("X_bench", X_bench.is_contiguous())
        print("X_train", X_train.is_contiguous())
        print("X_val", X_val.is_contiguous())
        print("X_test", X_test.is_contiguous())
        print("y_bench", y_bench.is_contiguous())
        print("y_train", y_train.is_contiguous())
        print("y_val", y_val.is_contiguous())
        print("y_test", y_test.is_contiguous())
        n_per_subset = len(X_train) // args.num_processes
        
        ## For each process, set up a loader.
        ## Each loader works with a view of a disjoint subset.
        data_loaders = []
        for i in range(args.num_processes):
            idx_start = i*n_per_subset
            idx_stop = idx_start + n_per_subset
            dl = DataLoader(
                dataset=TensorDataset(X_train[idx_start:idx_stop,:],
                                      y_train[idx_start:idx_stop]),
                **dl_kwargs
            )
            data_loaders.append(dl)
        print("Data prep complete.", "\n")
    
        ## Clerical points before model preparation.
        
        paras_dict = { pn: [] for pn in paras_todo }
        
        loss_fn, eval_fn_dict = get_eval(loss_fn=args.loss_fn,
                                         task_type=ds_paras["type"])

        ## First prepare the main worker models.
        print("Model prep time.")
        models = []
        for i in range(args.num_processes):
            model = Model_class(**ds_paras).to(dev)
            model.share_memory()
            models.append(model)
            for pn, p in model.named_parameters():
                if pn in paras_todo:
                    paras_dict[pn].append(p.clone().flatten())
                
        for pn in paras_todo:
            paras_dict[pn] = torch.stack(paras_dict[pn]).contiguous()
            print("paras_dict[pn].shape", paras_dict[pn].shape)
            print("paras_dict[pn]:")
            print(paras_dict[pn])

        
        ## Similarly, prepare the carrier model.
        model_carrier = Model_class(**ds_paras).to(dev)
    
        ## Also, prepare the benchmark learner.
        model_bench = Model_class(**ds_paras).to(dev)
        
        algo_bench = get_algo(algo_name=args.algo,
                              model=model_bench, **algo_kwargs)
        dl_bench = DataLoader(
            dataset=TensorDataset(X_bench, y_bench),
            **dl_kwargs
        )
        
        ## Now for the important swap (and remaining algo prep).
        algos = []
        for j, model in enumerate(models):
            for pn, p in model.named_parameters():
                if pn in paras_todo:
                    p.data = paras_dict[pn][j,:].view(p.shape)
            ## End loop over parameters, and append model.
            algo = get_algo(algo_name=args.algo,
                            model=model, **algo_kwargs)
            algos.append(algo)

    
        ## Before the main loop, prepare performance-storing arrays.
        losses_train = np.zeros(shape=(args.num_epochs, len(models)),
                                dtype=np.float32)
        losses_train_rb = np.zeros(shape=(args.num_epochs, len(todo_roboost)),
                                   dtype=np.float32)
        losses_test = None if X_test is None else np.zeros_like(losses_train)
        losses_test_rb = None if X_test is None else np.zeros_like(
            losses_train_rb
        )
        eval_train_dict = {
            key: np.zeros_like(losses_train) for key in eval_fn_dict.keys()
        }
        eval_train_dict_rb = {
            key: np.zeros_like(losses_train_rb) for key in eval_fn_dict.keys()
        }
        eval_test_dict = None if X_test is None else deepcopy(
            eval_train_dict
        )
        eval_test_dict_rb = None if X_test is None else deepcopy(
            eval_train_dict_rb
        )

    
        ## Loop over epochs is done in the parent process.
        for epoch in range(args.num_epochs):
            
            print("(Tr {}) Ep {} starting.".format(trial, epoch))
            
            ## First, run one epoch of the benchmark routine.
            train_epoch(num="bench",
                        model=model_bench,
                        algo=algo_bench,
                        data_loader=dl_bench,
                        loss_fn=loss_fn,
                        device=dev,
                        verbose=args.verbose)
            
            ## Next, zip up the worker models and loaders, and fire them up.
            zipped_train = zip(models, algos, data_loaders)
            procs = []
            for proc_num, (model, algo, dl) in enumerate(zipped_train):
                proc = ctx.Process(
                    target=train_epoch,
                    args=(proc_num, model, algo, dl, loss_fn, dev)
                )
                proc.start()
                print(
                    "(Tr {}) Ep {}. Proc {} started. is_alive = {}".format(
                        trial, epoch, proc_num, proc.is_alive()
                    )
                )
                procs.append(proc)
        
            ## Wait for all workers to finish before proceeding.
            for proc_num, proc in enumerate(procs):
                proc.join()
                print(
                    "(Tr {}) Ep {}. Proc {} joined. is_alive = {}".format(
                        trial, epoch, proc_num, proc.is_alive()
                    )
                )
            
            ## Print out parameters to check for a match.
            if args.verbose:
                print("Monitor after epoch {}".format(epoch))
                print("ACTUALS:")
                for model in models:
                    for pn, p in model.named_parameters():
                        print(pn, p)
                print("MONITOR:")
                for pn in paras_todo:
                    print(paras_dict[pn])
            
            ## Evaluate performance of the sub-process candidates.
            eval_models(epoch=epoch,
                        models=models,
                        loss_arrays=[losses_train, losses_test],
                        eval_dicts=[eval_train_dict, eval_test_dict],
                        data=(X_train,y_train,X_test,y_test),
                        device=dev,
                        loss_fn=loss_fn,
                        eval_fn_dict=eval_fn_dict)
            
            ## Evaluate performance of the roboost and benchmark methods.
            for j, rb in enumerate(todo_roboost):
                
                if todo_roboost[j] == "bench":
                    eval_model(epoch=epoch,
                               model=model_bench,
                               model_idx=j,
                               loss_arrays=[losses_train_rb,
                                            losses_test_rb],
                               eval_dicts=[eval_train_dict_rb,
                                           eval_test_dict_rb],
                               data=(X_train,y_train,X_test,y_test),
                               device=dev,
                               loss_fn=loss_fn,
                               eval_fn_dict=eval_fn_dict)
                else:
                    for pn, p in model_carrier.named_parameters():
                        if pn in paras_todo:
                            ## Modify the carrier model in place,
                            ## using the specified rb method.
                            do_roboost(para=p,
                                       ref_models=models,
                                       cand_tensor=paras_dict[pn],
                                       data_val=(X_val,y_val),
                                       device=dev,
                                       loss_fn=loss_fn,
                                       rb_method=rb,
                                       rg=rg_roboost)
                        else:
                            ## If a certain parameter is not to be
                            ## rb'd, then just choose a random candidate.
                            do_roboost(para=p,
                                       ref_models=models,
                                       cand_tensor=paras_dict[pn],
                                       data_val=(X_val,y_val),
                                       device=dev,
                                       loss_fn=loss_fn,
                                       rb_method="take-rand", # special here.
                                       rg=rg_roboost)
                    
                    ## After updating *all* relevant parameters,
                    ## we can now run the carrier through the
                    ## performance evaluation sub-routine.
                    eval_model(epoch=epoch,
                               model=model_carrier,
                               model_idx=j,
                               loss_arrays=[losses_train_rb,
                                            losses_test_rb],
                               eval_dicts=[eval_train_dict_rb,
                                           eval_test_dict_rb],
                               data=(X_train,y_train,X_test,y_test),
                               device=dev,
                               loss_fn=loss_fn,
                               eval_fn_dict=eval_fn_dict)
                    
            print("(Tr {}) Ep {} finished.".format(trial, epoch), "\n")
        
        ## Write performance to disk.
        perf_fname = os.path.join(towrite_dir,
                                  towrite_name+"-"+str(trial))
        eval_write(fname=perf_fname,
                   losses_train=losses_train,
                   losses_test=losses_test,
                   eval_train_dict=eval_train_dict,
                   eval_test_dict=eval_test_dict,
                   rb=False)
        eval_write(fname=perf_fname,
                   losses_train=losses_train_rb,
                   losses_test=losses_test_rb,
                   eval_train_dict=eval_train_dict_rb,
                   eval_test_dict=eval_test_dict_rb,
                   rb=True)

    ## Write a JSON file to disk that summarizes key experiment parameters.
    dict_to_json = vars(args)
    dict_to_json["todo_roboost"] = todo_roboost
    dict_to_json["chance_level"] = ds_paras["chance_level"]
    towrite_json = os.path.join(towrite_dir, towrite_name+".json")
    with open(towrite_json, "w", encoding="utf-8") as f:
        json.dump(obj=dict_to_json, fp=f,
                  ensure_ascii=False,
                  sort_keys=True, indent=4)


###############################################################################
