#!/bin/bash

source ~/ENV/bin/activate

game=${1:-"cheetah_run"}
steps=1000000

logdir="logs"

seed=1

test_args="agent=relo_sac use_per_buffer=True"
experiment="RELOSAC_baseline"

python train.py env=${game} seed=${seed} num_train_steps=${steps} experiment=${experiment} logdir=${logdir} ${test_args}
