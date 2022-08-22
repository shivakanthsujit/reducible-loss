#!/bin/bash

games="alien amidar assault bank_heist frostbite jamesbond seaquest"

for game in $games
do
    run_cmd="scripts/rainbow_exp.sh ${game}"
    run_cmd="scripts/relo_exp.sh ${game}"
    cmd="$run_cmd"
    echo -e "${cmd}"
    ${cmd}
    sleep 1
done