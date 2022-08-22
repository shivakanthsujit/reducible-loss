#!/bin/bash

games="cheetah_run walker_walk quadruped_walk quadruped_run walker_run finger_spin hopper_hop reacher_easy reacher_hard"

for game in $games
do
    run_cmd="scripts/exp.sh ${game}"
    run_cmd="scripts/per_exp.sh ${game}"
    run_cmd="scripts/relo_exp.sh ${game}"
    cmd="$run_cmd"
    echo -e "${cmd}"
    ${cmd}
    sleep 1
done