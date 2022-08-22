#!/bin/bash

games="asterix breakout freeway seaquest space_invaders"

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