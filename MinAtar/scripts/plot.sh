#!/bin/bash

source ~/ENV/bin/activate

labels="dqn_baseline Baseline per_dqn_baseline PER relo_dqn_baseline ReLo"

xlim=5000000
regex="baseline"
fname="minatar"
tasks='.*'
palette_choice=contrast
add="none"
cols=5

outdir=./plots/paper_plots

BINS=50000
logdir="logs"
yaxis="avg_return"
fname="baseline_${yaxis}"
python plotting.py --indir ${logdir} --tasks ${tasks} --methods ${regex} --labels ${labels} --outdir ${outdir} --fname ${fname} --xlabel Steps --ylabel Reward --xaxis step --yaxis ${yaxis} --add ${add} --bins ${BINS} --xlim 0 ${xlim} --ylimticks False --cols ${cols} --size 4 3.5 --palette ${palette_choice} --dpi 200

python make_iqm.py