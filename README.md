# sgd-roboost: robust confidence boosting of SGD sub-processes

In this repository, we provide software and demonstrations related to the following papers:

- Matthew J. Holland. Scaling-Up Robust Gradient Descent Techniques. *AAAI 2021*, to appear.
- Matthew J. Holland. <a href="https://arxiv.org/abs/2012.07346">Better scalability under potentially heavy-tailed feedback</a>. *Archival version*.

This project contains code which can be used to faithfully reproduce all the experimental results given in the above papers, and it can be easily applied to more general machine learning tasks outside the examples considered here.

At a high level, this repository is divided into two main parts:

- __Tests using real-world data:__ since we implement most of the back end using PyTorch, code related to the tests using benchmark data sets has a filename of the form `*_torch*`.
- __Tests using controlled simulations:__ all code related to simulation-centric experiments has a filename of the form `*_sims*`.

A table of contents for this README file:

- <a href="#data">Setup: preparing the benchmark data sets</a>
- <a href="#code_torch">Setup: software for tests using benchmark data</a>
- <a href="#code_sims">Setup: software for simulations</a>
- <a href="#demos">List of demos</a>

Before diving into any of the setup detailed below, we assume assume the user has access to a `bash` shell, can use `wget` to download data sets, has `git` and `conda` installed, and has run the following command:

```
$ conda update -n base conda
```

This repository contains code which is mostly quite "local" to the experiments carried out for this project, though it makes use of many functions which are of a much more general-purpose in nature. Such functions are implemented separately in our <a href="https://github.com/feedbackward/mml">mml repository</a> (details given below).

With that, let us get started on the data and software setup.


<a id="data"></a>
## Setup: preparing the benchmark data sets

Benchmark datasets can be acquired as follows.

```
$ git clone https://github.com/feedbackward/mml.git
$ conda create -n mml-data python=3.8 pip pytables scipy
$ conda activate mml-data
$ (mml-data) cd mml
$ (mml-data) pip install -e ./
```

Next, we need to specify where the data will be stored. There are two configuration variables in `mml/mml/config.py` to be set manually:

- `dir_data_toread`: this is where the raw data files will be stored.
- `dir_data_towrite`: this is where the processed data files (all in .h5 format) will be stored.

In particular, we will later be making use of the value assigned to `dir_data_towrite`; let us call this `DATADIR` for ease of reference later. Once these are set, `cd` to `mml/mml/data/`. From here, acquiring and processing the data is a one-line operation:

```
$ (mml-data) ./do_getdata.sh [adult ...]
```

This can be done whenever some dataset is desired. Furthermore, once all the desired datasets have been acquired, we can feel free to delete the `mml` directory we used here. One natural approach is to download raw data into `mml`, but set `DATADIR` to some other separate location to be used by other project-local software.


<a id="code_torch"></a>
## Setup: software for tests using benchmark data

We begin with software that is directly related to the demos of interest. We proceed assuming that the user is working in a directory that does *not* contain directories with the names of those we are about to `clone` (i.e., does __not__ contain `sgd-roboost` or `mml`), and that they have done all necessary preparations for installing PyTorch.

With these prerequisites understood and in place, the clerical setup is quite easy. The main initial steps are as follows.

```
$ git clone https://github.com/feedbackward/sgd-roboost.git
$ git clone https://github.com/feedbackward/mml.git
$ conda create -n sgd-roboost python=3.8 jupyter matplotlib pip pytables scipy
$ conda activate sgd-roboost
$ (sgd-roboost) conda install pytorch cudatoolkit=10.2 -c pytorch
$ (sgd-roboost) cd mml
$ (sgd-roboost) git checkout [SHA-1]
$ (sgd-roboost) pip install -e ./
$ (sgd-roboost) cd ../sgd-roboost
```

__Safe hash value:__ `2e88b08768f5db85ff2673fceca98078a07647a8` (tested 2020/12/23).

Next, we proceed under the assumption that the user has already completed the data acquisition as described in the previous section, i.e., the user has some `DATADIR` directory with all the processed `*.h5` files of interest.

With this preparation in place, we can move to running the experiments. First, various parameters related to the experiments are set as follows.

- __Data parameters:__ (in `setup_torch_data.py`)
  - Manually set the `dir_data_toread` variable to the `DATADIR` discussed in the previous section  (the default is literally called `DATADIR`, and is placed set in the home directory).
  - Dataset-specific training/validation set sizes are set using the parameters `n_train_frac` and `n_val_frac`. These are values between 0 and 1 that specify the fraction of the entire dataset to be used. The leftover data are used for testing.

- __Experiment parameters:__ (in `learn_torch_run.sh` and `learn_torch_run_*.sh`)
  - Parameters to be used commonly across all datasets are specified in `learn_torch_run.sh`.
  - Parameters that are specific to each data set are specified in the `learn_torch_run_*.sh` scripts.
  
- __Storage of results:__ results are stored in `[results_dir]/torch`, where `results_dir` is specified in `setup_results.py`.

- __Robust boosting settings:__ recalling "Merge" and "Valid" from the reference papers cited above, we consider several possibilities, all implemented and selected within `setup_torch_roboost.py`. The sub-routines to be tested are specified by the variable `todo_roboost`.

With all the parameters set as desired, execution is a one line operation.

```
./learn_torch_run.sh [dataset1 dataset2 ...]
```

The high level flow is as follows. All shell scripts are run using `bash`. The script `learn_torch_run.sh` saves the common parameters as environment variables, and goes to work executing `learn_torch_run_[dataset1].sh`, `learn_torch_run_[dataset2].sh`, and so on, in order. Within each script `learn_torch_run_*.sh`, the main Python script `learn_torch_driver.py` is called and passed all experiment parameters as arguments (both dataset-specific and dataset-independent parameters). Note that `learn_torch_driver.py` is called within a loop over what we call "tasks," where each task is characterized by one or more dataset-specific parameters and a task name, all of which are passed to the main driver script.

Results are written to disk with the following nomenclature: `[task]-[model]_[algo]-[trial].[descriptor]`.


<a id="code_sims"></a>
## Setup: software for simulations

*To be added no later than the end of January, 2021.*


<a id="demos"></a>
## List of demos

This repository includes detailed demonstrations to walk the user through re-creating the results in the papers cited at the top of this document. Below is a list of demo links which give our demos (originally in Jupyter notebook form) rendered using the useful <a href="https://github.com/jupyter/nbviewer">nbviewer</a> service.

- <a href="https://nbviewer.jupyter.org/github/feedbackward/sgd-roboost/blob/main/sgd-roboost/demo_torch.ipynb">Demo: tests using benchmark data</a>


