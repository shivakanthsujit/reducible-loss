import csv
import glob
import json
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import numpy as np
import scipy
import seaborn as sns
from rliable import library as rly
from rliable import metrics, plot_utils


# @title Plotting Helpers
def save_fig(fig, name):
    file_name = "{}.pdf".format(name)
    fig.savefig(file_name, format="pdf", bbox_inches="tight")
    files.download(file_name)
    return file_name


def set_axes(ax, xlim, ylim, xlabel, ylabel):
    ax.set_xlim(xlim)
    ax.set_ylim(ylim)
    ax.set_xlabel(xlabel, labelpad=14)
    ax.set_ylabel(ylabel, labelpad=14)


def set_ticks(ax, xticks, xticklabels, yticks, yticklabels):
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.set_yticks(yticks)
    ax.set_yticklabels(yticklabels)


def decorate_axis(ax, wrect=10, hrect=10, labelsize="large"):
    # Hide the right and top spines
    ax.spines["right"].set_visible(False)
    ax.spines["top"].set_visible(False)
    ax.spines["left"].set_linewidth(2)
    ax.spines["bottom"].set_linewidth(2)
    # Deal with ticks and the blank space at the origin
    ax.tick_params(length=0.1, width=0.1, labelsize=labelsize)
    # Pablos' comment
    ax.spines["left"].set_position(("outward", hrect))
    ax.spines["bottom"].set_position(("outward", wrect))


# @title Helpers for normalizing scores and plotting histogram plots.
def score_normalization(res_dict, min_scores, max_scores):
    games = res_dict.keys()
    norm_scores = {}
    for game, scores in res_dict.items():
        norm_scores[game] = (scores - min_scores[game]) / (
            max_scores[game] - min_scores[game]
        )
    return norm_scores


def convert_to_matrix(score_dict):
    keys = sorted(list(score_dict.keys()))
    return np.stack([score_dict[k] for k in keys], axis=1)


def plot_score_hist(
    score_matrix,
    bins=20,
    figsize=(28, 14),
    fontsize="xx-large",
    N=6,
    extra_row=1,
    names=None,
):
    num_tasks = score_matrix.shape[1]
    N1 = (num_tasks // N) + extra_row
    fig, ax = plt.subplots(nrows=N1, ncols=N, figsize=figsize)
    for i in range(N):
        for j in range(N1):
            idx = j * N + i
            if idx < num_tasks:
                ax[j, i].set_title(names[idx], fontsize=fontsize)
                sns.histplot(score_matrix[:, idx], bins=bins, ax=ax[j, i], kde=True)
            else:
                ax[j, i].axis("off")
            decorate_axis(ax[j, i], wrect=5, hrect=5, labelsize="xx-large")
            ax[j, i].xaxis.set_major_locator(plt.MaxNLocator(4))
            if idx % N == 0:
                ax[j, i].set_ylabel("Count", size=fontsize)
            else:
                ax[j, i].yaxis.label.set_visible(False)
            ax[j, i].grid(axis="y", alpha=0.1)
    return fig


# @title Stratified Bootstrap CIs and Aggregate metrics

StratifiedBootstrap = rly.StratifiedBootstrap

IQM = lambda x: metrics.aggregate_iqm(x)  # Interquartile Mean
OG = lambda x: metrics.aggregate_optimality_gap(x, 1.0)  # Optimality Gap
MEAN = lambda x: metrics.aggregate_mean(x)
MEDIAN = lambda x: metrics.aggregate_median(x)

files = glob.glob("logs/**/metrics.jsonl", recursive=True)
yaxis = "eval/episode_reward"
data = defaultdict(lambda: defaultdict(list))
for fname in files:
    with open(fname, "r") as f:
        json_list = list(f)
    for json_str in json_list:
        if yaxis in json_str:
            result = json.loads(json_str)
            step = result.pop("step")
    parts = fname.split("/")
    seed = parts[-2]
    run = parts[-3]
    game = parts[-4]
    data[run][game].append(result[yaxis])

# @title Load DMC data
DMC_ENVS = sorted(
    [
        "cheetah_run",
        "finger_spin",
        "hopper_hop",
        "quadruped_run",
        "quadruped_walk",
        "reacher_easy",
        "reacher_hard",
        "walker_run",
        "walker_walk",
    ]
)


def cut_data(x, seeds, envs):
    x = {k: x[k][:seeds] for k in x.keys()}
    return {k: x[k] for k in envs}


outdir = "plots"
seeds = 5
algs = ["RELOSAC_baseline", "PERSAC_baseline", "baseline"]
dmc_scores = {
    alg: convert_to_matrix(cut_data(data[alg], seeds, DMC_ENVS)) for alg in algs
}
normalized_dmc_scores = {alg: scores / 1000 for alg, scores in dmc_scores.items()}

import pickle

with open("tmp3/dmc.pkl", "wb") as f:
    pickle.dump(normalized_dmc_scores, f)


def make_csv_data(scores):
    games_header = [[f"{game}_mean", f"{game}_std"] for game in DMC_ENVS]
    games_header = np.array(games_header).flatten()
    scores_header = ["alg"] + games_header.tolist()
    scores_stats = {k: [v.mean(0), v.std(0)] for k, v in scores.items()}
    scores_stats = {
        k: np.array([[m, s] for m, s in zip(*v)]) for k, v in scores_stats.items()
    }
    scores_stats = {k: v.flatten() for k, v in scores_stats.items()}
    scores_stats_data = [[k] + v.tolist() for k, v in scores_stats.items()]
    scores_stats_csv = [scores_header] + scores_stats_data
    return scores_stats_csv


def write_csv(csv_data, fname):
    with open(fname, "w", newline="") as myfile:
        wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
        wr.writerows(csv_data)
    print(f"Saving csv to {fname}")


mean_scores_csv = make_csv_data(dmc_scores)
fname = os.path.join(outdir, "mean_scores.csv")
write_csv(mean_scores_csv, fname)

mean_norm_scores_csv = make_csv_data(normalized_dmc_scores)
fname = os.path.join(outdir, "mean_norm_scores.csv")
write_csv(mean_norm_scores_csv, fname)

# @title setup colors

colors = sns.color_palette("colorblind")
color_idxs = [0, 3, 4, 2, 1] + list(range(9, 4, -1))
DMC_COLOR_DICT = dict(zip(algs, [colors[idx] for idx in color_idxs]))

# @title Calculate bootstrap CIs for mean scores

mean_func = lambda x: np.array([MEAN(x)])
score_dmc_all, all_mean_CIs = rly.get_interval_estimates(
    normalized_dmc_scores, mean_func, reps=50000
)

# @title Plot Mean Scores

legend = {
    "RELOSAC_baseline": "ReLo",
    "PERSAC_baseline": "PER",
    "baseline": "Baseline",
}

# DMC_COLOR_DICT = dict(zip(algs, [colors[idx] for idx in color_idxs]))
DMC_COLOR_DICT = dict(zip(list(legend.values()), [colors[idx] for idx in color_idxs]))

# @title Aggregates on DMC (with 10 runs)
aggregate_func = lambda x: np.array([MEDIAN(x), IQM(x), MEAN(x), OG(x)])
aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
    normalized_dmc_scores, aggregate_func, reps=50000
)
scoreswlegend = {legend[k]: v for k, v in aggregate_scores.items()}
intervalswlegend = {legend[k]: v for k, v in aggregate_interval_estimates.items()}
fig, axes = plot_utils.plot_interval_estimates(
    scoreswlegend,
    intervalswlegend,
    metric_names=["Median", "IQM", "Mean", "Optimality Gap"],
    algorithms=list(legend.values()),
    colors=DMC_COLOR_DICT,
    xlabel_y_coordinate=-0.5,
    xlabel="Max Normalized Score",
)
fname = os.path.join(outdir, "agg.jpg")
print(f"Saving plot to {fname}")
plt.savefig(fname, dpi=120, bbox_inches="tight")
plt.close()


aggregate_func = lambda x: np.array([MEDIAN(x), MEAN(x)])
aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
    normalized_dmc_scores, aggregate_func, reps=50000
)
scoreswlegend = {legend[k]: v for k, v in aggregate_scores.items()}
intervalswlegend = {legend[k]: v for k, v in aggregate_interval_estimates.items()}
fig, axes = plot_utils.plot_interval_estimates(
    scoreswlegend,
    intervalswlegend,
    metric_names=["Median", "Mean"],
    algorithms=list(legend.values()),
    colors=DMC_COLOR_DICT,
    xlabel_y_coordinate=-0.5,
    xlabel="Max Normalized Score",
)
fname = os.path.join(outdir, "agg_mean_median.jpg")
print(f"Saving plot to {fname}")
plt.savefig(fname, dpi=120, bbox_inches="tight")
plt.close()


aggregate_func = lambda x: np.array([IQM(x), OG(x)])
aggregate_scores, aggregate_interval_estimates = rly.get_interval_estimates(
    normalized_dmc_scores, aggregate_func, reps=50000
)
scoreswlegend = {legend[k]: v for k, v in aggregate_scores.items()}
intervalswlegend = {legend[k]: v for k, v in aggregate_interval_estimates.items()}
fig, axes = plot_utils.plot_interval_estimates(
    scoreswlegend,
    intervalswlegend,
    metric_names=["IQM", "Optimality Gap"],
    algorithms=list(legend.values()),
    colors=DMC_COLOR_DICT,
    xlabel_y_coordinate=-0.5,
    xlabel="Max Normalized Score",
)
fname = os.path.join(outdir, "agg_iqm.jpg")
print(f"Saving plot to {fname}")
plt.savefig(fname, dpi=120, bbox_inches="tight")
plt.close()
