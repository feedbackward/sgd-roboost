#!/bin/bash

## Dataset-specific arguments to be passed.
DATA="cifar10"
PROCS="20"
TASKS=("low" "lowmed" "medhigh" "high")
STEPS=("0.0025" "0.005" "0.01" "0.02")

for idx in "${!TASKS[@]}"
do python "learn_torch_driver.py" --algo="$ALGO" --batch-size="$BATCH" --data="$DATA" --loss-fn="$LOSSFN" --model="$MODEL" --num-epochs="$EPOCHS" --num-processes="$PROCS" --num-trials="$TRIALS" --step-size="${STEPS[idx]}" --task-name="${TASKS[idx]}"
done

