#!/bin/bash

source ~/ENV/bin/activate

game=${1:-"breakout"}
steps=2000000

# * Normal PER Replay
ID="per_replay"
test_args=""

ID="${ID}_2M"

logdir="logs"
seed=1
python main.py --id ${ID} --logdir ${logdir} --game ${game} --T-max ${steps} --seed ${seed} ${test_args}
