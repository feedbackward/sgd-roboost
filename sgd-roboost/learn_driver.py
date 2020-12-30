'''Driver script for tests of robust confidence boosting methods.'''

## External modules.
import argparse
import json
import numpy as np
import os

## Internal modules.
from mml.utils import makedir_safe
from setup_algos import get_algo
from setup_data import get_data
from setup_eval import get_eval, eval_model, eval_models, eval_write
from setup_inits import get_w_init
from setup_losses import get_loss
from setup_models import get_model
from setup_results import results_dir
from setup_roboost import do_roboost, todo_roboost
from setup_train import train_epoch


###############################################################################


## Basic setup.

parser = argparse.ArgumentParser(description="Arguments for driver script.")

parser.add_argument("--algo",
                    help="Algorithm class. (default: SGD)",
                    type=str, default="SGD", metavar="S")
parser.add_argument("--batch-size",
                    help="Mini-batch size for algorithms (default: 1).",
                    type=int, default=1, metavar="N")
parser.add_argument("--data",
                    help="Specify data set to be used (default: None).",
                    type=str, default=None, metavar="S")
parser.add_argument("--loss",
                    help="Loss name. (default: quadratic)",
                    type=str, default="quadratic", metavar="S")
parser.add_argument("--model",
                    help="Model class. (default: linreg)",
                    type=str, default="linreg", metavar="S")
parser.add_argument("--num-epochs",
                    help="Number of epochs to run (default: 3)",
                    type=int, default=3, metavar="N")
parser.add_argument("--num-processes",
                    help="Number of learning sub-processes (default: 1)",
                    type=int, default=1, metavar="N")
parser.add_argument("--num-trials",
                    help="Number of independent random trials (default: 1)",
                    type=int, default=1, metavar="N")
parser.add_argument("--step-size",
                    help="Step size parameter (default: 0.01)",
                    type=float, default=0.01, metavar="F")
parser.add_argument("--task-name",
                    help="A task name. Default is the word default.",
                    type=str, default="default", metavar="S")
parser.add_argument("--verbose",
                    help="Print details or not (default: False).",
                    action="store_true", default=False)


## Setup of random generator.
ss = np.random.SeedSequence()
rg = np.random.default_rng(seed=ss)

## Parse the arguments passed via command line.
args = parser.parse_args()
if args.data is None:
    raise TypeError("Given --data=None, should be a string.")

## Name to be used identifying the results etc. of this experiment.
towrite_name = args.task_name+"-"+"_".join([args.model, args.algo])

## Model class must be initialized here, to ensure all sub-procs get access.
Model_class = get_model(model_class=args.model)

## Prepare a directory to save results.
towrite_dir = os.path.join(results_dir, "sims", args.data)
makedir_safe(towrite_dir)


## Main process.
if __name__ == "__main__":
    
    ## Prepare the loss for training.
    loss = get_loss(name=args.loss)

    ## Arguments for algorithms.
    algo_kwargs = {"step_size": args.step_size}
    
    ## Arguments for models.
    model_kwargs = {}
    
    ## Start the loop over independent trials.
    for trial in range(args.num_trials):
        
        ## Load in data.
        print("Start data prep.")
        (X_train, y_train, X_val, y_val,
         X_test, y_test, ds_paras) = get_data(dataset=args.data,
                                              rg=rg)
        n_per_subset = len(X_train) // args.num_processes
        
        ## Indices to evenly split the training data (toss the excess).
        data_indices = []
        for i in range(args.num_processes):
            idx_start = i*n_per_subset
            idx_stop = idx_start + n_per_subset
            data_idx = np.arange(start=idx_start, stop=idx_stop)
            data_indices.append(data_idx)
        print("Data prep complete.", "\n")

        ## Prepare evaluation metric(s).
        eval_dict = get_eval(loss_name=args.loss,
                             model_name=args.model, **ds_paras)

        ## First randomly initialize the parameters for each model.
        cand_array = []
        for i in range(args.num_processes):
            cand_array.append(get_w_init(rg=rg, **ds_paras))
        cand_array = np.hstack(cand_array).T
        
        ## Next initialize the models with views of the parameters.
        models = []
        for i in range(len(cand_array)):
            model = Model_class(w_init=cand_array[i:(i+1),:].T)
            models.append(model)
        
        ## Prepare the carrier model.
        model_carrier = Model_class(w_init=get_w_init(rg=rg, **ds_paras))
        
        ## Prepare algorithms.
        algos = []
        for j, model in enumerate(models):
            algo = get_algo(name=args.algo,
                            model=model,
                            loss=loss,
                            **ds_paras, **algo_kwargs)
            algos.append(algo)
        
        ## Prepare storage for performance evaluation this trial.
        store_train = {
            key: np.zeros(shape=(args.num_epochs, len(models)),
                          dtype=np.float32) for key in eval_dict.keys()
        }
        store_train_rb = {
            key: np.zeros(shape=(args.num_epochs, len(todo_roboost)),
                          dtype=np.float32) for key in eval_dict.keys()
        }
        if X_test is not None:
            store_test = {
                key: np.zeros_like(
                    store_train[key]
                ) for key in eval_dict.keys()
            }
            store_test_rb = {
                key: np.zeros_like(
                    store_train_rb[key]
                ) for key in eval_dict.keys()
            }
        else:
            store_test = {}
            store_test_rb = {}
        storage = (store_train, store_test)
        storage_rb = (store_train_rb, store_test_rb)
        
        ## Loop over epochs, done in the parent process.
        for epoch in range(args.num_epochs):
            
            print("(Tr {}) Ep {} starting.".format(trial, epoch))
            
            ## Shuffle within subsets.
            for data_idx in data_indices:
                rg.shuffle(data_idx)

            ## Zip up the worker elements, and put them to work.
            zipped_train = zip(algos, data_indices)
            for proc_num, (algo, data_idx) in enumerate(zipped_train):
                if args.verbose:
                    print(
                        "(Tr {}) Ep {}. Proc {} started.".format(
                            trial, epoch, proc_num
                        )
                    )
                train_epoch(num=proc_num,
                            algo=algo,
                            loss=loss,
                            X=X_train[data_idx,:],
                            y=y_train[data_idx,:],
                            batch_size=args.batch_size)
                if args.verbose:
                    print(
                        "(Tr {}) Ep {}. Proc {} finished.".format(
                            trial, epoch, proc_num
                        )
                    )
                
            ## Evaluate performance of the sub-process candidates.
            eval_models(epoch=epoch,
                        models=models,
                        storage=storage,
                        data=(X_train,y_train,X_test,y_test),
                        eval_dict=eval_dict)
            
            ## Do robust boosting and evaluate performance.
            for j, rb in enumerate(todo_roboost):

                do_roboost(model_todo=model_carrier,
                           ref_models=models,
                           cand_array=cand_array,
                           data_val=(X_val,y_val),
                           loss=loss,
                           rb_method=rb,
                           rg=rg)

                eval_model(epoch=epoch,
                           model=model_carrier,
                           model_idx=j,
                           storage=storage_rb,
                           data=(X_train,y_train,X_test,y_test),
                           eval_dict=eval_dict)

            print("(Tr {}) Ep {} finished.".format(trial, epoch), "\n")


        ## Write performance for this trial to disk.
        perf_fname = os.path.join(towrite_dir,
                                  towrite_name+"-"+str(trial))
        eval_write(fname=perf_fname,
                   storage=storage,
                   rb=False)
        eval_write(fname=perf_fname,
                   storage=storage_rb,
                   rb=True)

    ## Write a JSON file to disk that summarizes key experiment parameters.
    dict_to_json = vars(args)
    dict_to_json.update({
        "entropy": ss.entropy, # for reproducability.
        "todo_roboost": todo_roboost # for reference.
    })
    towrite_json = os.path.join(towrite_dir, towrite_name+".json")
    with open(towrite_json, "w", encoding="utf-8") as f:
        json.dump(obj=dict_to_json, fp=f,
                  ensure_ascii=False,
                  sort_keys=True, indent=4)


###############################################################################

