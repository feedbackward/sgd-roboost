#!/bin/bash

ALGO="SGD"
BATCH="1"
DATA="ds_lognormal"
LOSS="quadratic"
MODEL="linreg"
EPOCHS="40"
TASK="default"
TRIALS="100"
PROCS="10"
STEP="0.01"

python "learn_driver.py" --algo="$ALGO" --batch-size="$BATCH" --data="$DATA" --loss="$LOSS" --model="$MODEL" --num-epochs="$EPOCHS" --num-processes="$PROCS" --num-trials="$TRIALS" --step-size="$STEP" --task-name="$TASK"

