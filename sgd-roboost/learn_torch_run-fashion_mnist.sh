#!/bin/bash

## Dataset-specific arguments to be passed.
DATA="fashion_mnist"
PROCS="20"
TASKS=("low" "lowmed" "medhigh" "high")
STEPS=("0.025" "0.05" "0.1" "0.2")

for idx in "${!TASKS[@]}"
do python "learn_torch_driver.py" --algo="$ALGO" --batch-size="$BATCH" --data="$DATA" --loss-fn="$LOSSFN" --model="$MODEL" --num-epochs="$EPOCHS" --num-processes="$PROCS" --num-trials="$TRIALS" --step-size="${STEPS[idx]}" --task-name="${TASKS[idx]}"
done

