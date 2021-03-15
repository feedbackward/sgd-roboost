# sgd-roboost: robust confidence boosting of SGD sub-processes

In this repository, we provide software and demonstrations related to the following papers:

- Robustness and scalability under heavy tails, without strong convexity. Matthew J. Holland. *AISTATS 2021*.
- Scaling-Up Robust Gradient Descent Techniques. Matthew J. Holland. *AAAI 2021*.
- <a href="https://arxiv.org/abs/2012.07346">Better scalability under potentially heavy-tailed feedback</a>. Matthew J. Holland. *Archival version*.

This repository contains code which can be used to faithfully reproduce all the experimental results given in the above papers, and it can be easily applied to more general machine learning tasks outside the examples considered here.

At a high level, this repository is divided into two main parts:

- __Tests using real-world data:__ since we implement most of the back end using PyTorch, code specifically related to the tests using benchmark data sets has filenames of the form `learn_torch_*` and `setup_torch_*.py`.
- __Tests using controlled simulations:__ all other files of the form `learn_*` or `setup_*.py` without the `torch` substring is related to our simulation-based tests.

A table of contents for this README file:

- <a href="#setup_init">Setup: initial software preparation</a>
- <a href="#setup_data">Setup: preparing the benchmark data sets</a>
- <a href="#setup_torch">Setup: tests using PyTorch</a>
- <a href="#setup_sims">Setup: simulation-based tests using Numpy</a>
- <a href="#demos">List of demos</a>
- <a href="#safehash">Safe hash values</a>


<a id="setup_init"></a>
## Setup: initial software preparation

Before getting into the specifics of using this repository, please carry out the following preparatory steps.

- Ensure you have the <a href="https://github.com/feedbackward/mml#prereq">prerequisite software</a> used in the setup of our `mml` repository.
- Ensure that you can install and use <a href="https://pytorch.org/">PyTorch</a>on your system. Furthermore, if you wish to use the <a href="https://developer.nvidia.com/cuda-toolkit">CUDA Toolkit</a>, it is already installed.

Next, make a local copy of the repository and create a virtual environment for working in as follows:

```
$ git clone https://github.com/feedbackward/sgd-roboost.git
$ conda create -n sgd-roboost python=3.8 jupyter matplotlib pip pytables scipy
$ conda activate sgd-roboost
```

Having made (and activated) this new environment, we would like to use `pip` to install supporting software for convenient access. This is done easily, by simply running

```
(sgd-roboost) $ cd [mml path]/mml
(sgd-roboost) $ pip install -e
```

and filling in the path placeholder with the path to wherever you used `clone` to copy the `mml` repository to. If you desire a safe, tested version of `mml`, just run

```
(sgd-roboost) $ git checkout [safe hash mml]
```

before the the `pip install -e ./` command given above, and replacing the safe hash placeholder according to the latest  <a href="#safehash">safe hash value</a> documented below.

To install PyTorch (within the `sgd-roboost` virtual environment), run

```
(sgd-roboost) $ conda install pytorch cudatoolkit=[CUDA version] -c pytorch -c conda-forge
```

being sure to replace `[CUDA version]` with the appropriate CUDA toolkit version depending on your hardware (if applicable; see <a href="https://pytorch.org/get-started/locally/">theofficial PyTorch site</a> for more details).


<a id="setup_data"></a>
## Setup: preparing the benchmark data sets

Please follow the instructions under <a href="https://github.com/feedbackward/mml#data">"Acquiring benchmark datasets"</a> using our `mml` repository. The rest of this README assumes that the user has prepared any desired benchmark datasets, stored in a local data storage directory (default path is `[path to mml]/mml/mml/data` as specified by the variable `dir_data_towrite` in `mml/mml/config.py`. Henceforth, we will refer to the directory housing the HDF5-formatted data sub-directories as __Data-Main__.


<a id="setup_torch"></a>
## Setup: tests using PyTorch

We can now start with configuration and execution of our PyTorch-based numerical experiments. First, various parameters related to the experiments are set as follows.

- __Data parameters:__ (in `setup_torch_data.py`)
  - Manually set the `dir_data_toread` variable to __Data-Main__ discussed in the previous section.
  - Dataset-specific training/validation set sizes are set using the parameters `n_train_frac` and `n_val_frac` set to default values specified in `mml/mml/data/__init__.py`. These are values between 0 and 1 that specify the fraction of the entire dataset to be used. All leftover data is used for testing.

- __Experiment parameters:__ (in `learn_torch_run.sh` and `learn_torch_run_*.sh`)
  - Parameters to be used commonly across all datasets are specified in `learn_torch_run.sh`.
  - Parameters that are specific to each data set are specified in the `learn_torch_run_*.sh` scripts.
  
- __Storage of results:__ results are stored in `[results_dir]/torch`, where `results_dir` is specified in `setup_results.py`.

- __Robust boosting settings:__ recalling "Merge" and "Valid" from the reference papers cited above, we consider several possibilities, all implemented and selected within `setup_torch_roboost.py`. The sub-routines to be tested are specified by the variable `todo_roboost`.

With all the parameters set as desired, execution is a one line operation.

```
(sgd-roboost) $ bash learn_torch_run.sh [dataset1 dataset2 ...]
```

The high level flow is as follows. All shell scripts are run using `bash`. The script `learn_torch_run.sh` saves the common parameters as environment variables, and goes to work executing `learn_torch_run_[dataset1].sh`, `learn_torch_run_[dataset2].sh`, and so on, in order. Within each script `learn_torch_run_*.sh`, the main Python script `learn_torch_driver.py` is called and passed all experiment parameters as arguments (both dataset-specific and dataset-independent parameters). Note that `learn_torch_driver.py` is called within a loop over what we call "tasks," where each task is characterized by one or more dataset-specific parameters and a task name, all of which are passed to the main driver script.

Results are written to disk with the following nomenclature: `[task]-[model]_[algo]-[trial].[descriptor]`.


<a id="setup_sims"></a>
## Setup: simulation-based tests using Numpy

Here we describe how to run the simulation-based experiments that run purely on Numpy and SciPy, and do not make use of PyTorch. The key experimental parameters (and clerical settings) are specified in the following files.

- __Data parameters:__ (in `setup_data.py`) everything from sample size and underlying model dimension to noise distributions is specified within this file.

- __Experiment parameters:__ (in `learn_run.sh`) the choice of algorithm, model, data generation protocol, among other key parameters is made within this simple shell script.
  
- __Storage of results:__ results are stored by default `[results_dir]/sims`, where `results_dir` is specified in `setup_results.py`.

- __Robust boosting settings:__ all specified within `setup_roboost.py`. The sub-routines to be tested are specified by the variable `todo_roboost`.

The overall flow is quite straightforward. Running the script `learn_run.sh` in `bash`, the main Python driver script `learn_driver.py` is passed all the key experimental parameters as arguments.


<a id="demos"></a>
## List of demos

This repository includes detailed demonstrations to walk the user through re-creating the results in the papers cited at the top of this document. Below is a list of demo links which give our demos (originally in Jupyter notebook form) rendered using the useful <a href="https://github.com/jupyter/nbviewer">nbviewer</a> service.

- <a href="https://nbviewer.jupyter.org/github/feedbackward/sgd-roboost/blob/main/roboost/demo_torch.ipynb">Demo: tests using benchmark data</a>
- <a href="https://nbviewer.jupyter.org/github/feedbackward/sgd-roboost/blob/main/roboost/demo_sims.ipynb">Demo: tests using simulations</a>


<a id="safehash"></a>
## Safe hash values

- Replace `[safe hash mml]` with `1f6fa730e86ad7da88ba5b33400b0ec476d2cd1d`.

__Date of safe hash test:__ 2021/03/12.
