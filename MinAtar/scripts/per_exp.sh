#!/bin/bash

source ~/ENV/bin/activate

environment_name=${1:-"breakout"}

logdir="logs"

seed=1

ID="baseline"

python examples/per_dqn.py --game ${environment_name} --id ${ID} --logdir ${logdir} --seed ${seed} --save
