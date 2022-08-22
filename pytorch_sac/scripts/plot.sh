#!/bin/bash

source ~/ENV/bin/activate

labels="baseline Baseline PERSAC_baseline PER RELOSAC_baseline ReLo"

tasks="cheetah_run finger_spin hopper_hop quadruped_run quadruped_walk reacher_easy reacher_hard walker_run walker_walk"

xlim=1000000
regex="^baseline\$ PERSAC_baseline\$ RELOSAC_baseline\$"
fname="dmc_full"
palette_choice=contrast
logdir="logs"
add="none"
cols=3
outdir="./plots"
yaxis="eval/episode_reward"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ${outdir} --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins 5000 --xlim 0 ${xlim} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200

python make_iqm.py
