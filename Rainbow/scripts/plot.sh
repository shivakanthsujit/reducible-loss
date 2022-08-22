#!/bin/bash

source ~/ENV/bin/activate

xlim=2000000
labels="${labels} per_replay_2M Rainbow"
labels="${labels} relo_priority_2M Rainbow+ReLo"
regex="2M"
fname="atari_2M"
tasks='.*'
tasks="alien amidar assault bank_heist frostbite jamesbond seaquest"
palette_choice=contrast
logdir="logs"
add="none"
cols=4
colors="per_replay_2M #33aa00 relo_priority_2M #ff0011"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ./plots --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis avg_reward --add ${add} --bins 50000 --xlim 0 ${xlim} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --colors ${colors} --dpi 200

python make_iqm.py