#!/bin/bash

## Arguments common across all settings, to be set as environment vars.
## Ref: https://stackoverflow.com/a/28490273
set -a
ALGO="SGD"
BATCH="8"
LOSSFN="nll"
MODEL="FF_L3"
EPOCHS="15"
TRIALS="25"
set +a

## A simple loop over all the datasets specified.
for arg
do bash "learn_torch_run-${arg}.sh"
done
