#!/bin/bash

## Dataset-specific arguments to be passed.
DATA="cod_rna"
PROCS="20"
TASKS=("low" "lowmed" "medhigh" "high")
STEPS=("0.15" "0.3" "0.6" "1.2")

for idx in "${!TASKS[@]}"
do python "learn_torch_driver.py" --algo="$ALGO" --batch-size="$BATCH" --data="$DATA" --loss-fn="$LOSSFN" --model="$MODEL" --num-epochs="$EPOCHS" --num-processes="$PROCS" --num-trials="$TRIALS" --step-size="${STEPS[idx]}" --task-name="${TASKS[idx]}"
done

