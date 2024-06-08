import os
import pickle
import random
from typing import List, Dict, Union

import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import wandb
from tqdm import tqdm

os.makedirs("out", exist_ok=True)
os.makedirs("bin", exist_ok=True)

plt.rcParams['figure.figsize'] = (10, 6)
plt.rcParams['figure.dpi'] = 300
sns.set(style="ticks", font_scale=1.5)
plt.rcParams.update({
    # 'font.family': 'serif',
    'font.serif': 'Times New Roman'
})

# # EOP

"""
Taken from here and a bit modified
https://github.com/dodgejesse/show_your_work/
"""


def _cdf_with_replacement(i, n, N):
    return (i / N) ** n


def _compute_stds(N, cur_data, expected_max_cond_n, pdfs):
    """
    this computes the standard error of the max.
    this is what the std dev of the bootstrap estimates of the mean of the max converges to, as
    is stated in the last sentence of the summary on page 10 of 
    http://www.stat.cmu.edu/~larry/=stat705/Lecture13.pdf
    """
    std_of_max_cond_n = []
    for n in range(N):
        # for a given n, estimate variance with \sum(p(x) * (x-mu)^2), where mu is \sum(p(x) * x).
        cur_std = 0
        for i in range(N):
            cur_std += (cur_data[i] - expected_max_cond_n[n]) ** 2 * pdfs[n][i]
        cur_std = np.sqrt(cur_std)
        std_of_max_cond_n.append(cur_std)
    return std_of_max_cond_n


# this implementation assumes sampling with replacement for computing the empirical cdf
def expected_online_performance(
        online_performance: List[float],
        output_n: int
) -> Dict[str, Union[List[float], float]]:
    # Copy and sort?
    online_performance = list(online_performance)
    online_performance.sort()

    N = len(online_performance)
    pdfs = []
    for n in range(1, N + 1):
        # the CDF of the max
        F_Y_of_y = []
        for i in range(1, N + 1):
            F_Y_of_y.append(_cdf_with_replacement(i, n, N))

        f_Y_of_y = []
        cur_cdf_val = 0
        for i in range(len(F_Y_of_y)):
            f_Y_of_y.append(F_Y_of_y[i] - cur_cdf_val)
            cur_cdf_val = F_Y_of_y[i]

        pdfs.append(f_Y_of_y)

    expected_max_cond_n = []
    expected_med_cond_n = []
    expected_iqr_cond_n = []
    for n in range(N):
        # for a given n, estimate expected value with \sum(x * p(x)), where p(x) is prob x is max.
        cur_expected = 0
        for i in range(N):
            cur_expected += online_performance[i] * pdfs[n][i]
        expected_max_cond_n.append(cur_expected)

        # estimate median
        cur_sum = 0.0
        for i in range(N):
            cur_sum += pdfs[n][i]
            if cur_sum == 0.5:
                expected_med_cond_n.append(online_performance[i])
                break
            elif cur_sum > 0.5:
                # nearest strat
                cur_diff = cur_sum - 0.5
                prev_dif = 0.5 - (cur_sum - pdfs[n][-1])
                if cur_diff < prev_dif:
                    expected_med_cond_n.append(online_performance[i])
                else:
                    expected_med_cond_n.append(online_performance[i - 1])
                break

        # estimate iqr
        cur_sum = 0.0
        percent25 = 0.0
        checked25 = False

        percent75 = 0.0
        checked75 = False
        for i in range(N):
            cur_sum += pdfs[n][i]
            if not checked25:
                if cur_sum == 0.25:
                    percent25 = online_performance[i]
                    checked25 = True
                elif cur_sum > 0.25:
                    # nearest strat
                    cur_diff = cur_sum - 0.25
                    prev_dif = 0.25 - (cur_sum - pdfs[n][-1])
                    if cur_diff < prev_dif:
                        percent25 = online_performance[i]
                    else:
                        percent25 = online_performance[i - 1]

            if not checked75:
                if cur_sum == 0.75:
                    percent75 = online_performance[i]
                    checked75 = True
                elif cur_sum > 0.75:
                    # nearest strat
                    cur_diff = cur_sum - 0.75
                    prev_dif = 0.75 - (cur_sum - pdfs[n][-1])
                    if cur_diff < prev_dif:
                        percent75 = online_performance[i]
                    else:
                        percent75 = online_performance[i - 1]
        expected_iqr_cond_n.append(percent75 - percent25)

    std_of_max_cond_n = _compute_stds(N, online_performance, expected_max_cond_n, pdfs)

    return {
        "median": expected_med_cond_n[:output_n],
        "iqr": expected_iqr_cond_n[:output_n],
        "mean": expected_max_cond_n[:output_n],
        "std": std_of_max_cond_n[:output_n],
        "max": np.max(online_performance),
        "min": np.min(online_performance)
    }


def expected_online_performance_arbit(
        online_performance: List[float],
        offline_performance: List[float],
        output_n: int
) -> Dict[str, Union[List[float], float]]:
    means = [x for _, x in sorted(zip(offline_performance, online_performance), key=lambda pair: pair[0], reverse=True)]

    if len(means) > 0:
        cur_max = means[0]
        for ind in range(len(means)):
            cur_max = max(cur_max, means[ind])
            means[ind] = cur_max

    return {
        "mean": means[:output_n],
        "std": means[:output_n],
        "max": np.max(online_performance),
        "min": np.min(online_performance)
    }


def get_data_from_sweeps(sweeps_ids, param_1="actor_bc_coef", param_2="critic_bc_coef", param_3=None):
    maxes = []
    lasts = []
    name_list = []
    config_list = []
    full_scores = {}

    for s in tqdm(sweeps_ids, desc="Sweeps processing", position=0, leave=True):
        api = wandb.Api(timeout=39)
        sweep = api.sweep(s)
        runs = sweep.runs
        cur_max = 0
        for run in tqdm(runs, desc="Runs processing", position=0, leave=True):
            all_scores = []

            config = {k: v for k, v in run.config.items() if not k.startswith('_')}
            # print(run.name, end=' ')
            for i, row in run.history(keys=["eval/normalized_score_mean"], samples=3000).iterrows():
                last = row["eval/normalized_score_mean"]
                all_scores.append(last)
            cur_max = max(cur_max, len(all_scores))
            if len(all_scores) < 100 and "antmaze" not in config["dataset_name"]:
                all_scores = [0] * cur_max
            if config["dataset_name"] not in full_scores:
                full_scores[config["dataset_name"]] = {}
            if str(config[param_1]) not in full_scores[config["dataset_name"]]:
                full_scores[config["dataset_name"]][str(config[param_1])] = {}
            if str(config[param_2]) not in full_scores[config["dataset_name"]][str(config[param_1])]:
                if param_3 is None:
                    full_scores[config["dataset_name"]][str(config[param_1])][str(config[param_2])] = []
                else:
                    full_scores[config["dataset_name"]][str(config[param_1])][str(config[param_2])] = {}
            if param_3 is not None and str(config[param_3]) not in full_scores[config["dataset_name"]][str(config[param_1])][str(config[param_2])]:
                full_scores[config["dataset_name"]][str(config[param_1])][str(config[param_2])][str(config[param_3])] = []
            # print("LEN", len(all_scores))
            if len(all_scores) == 0:
                continue
            last_score_idx = -1
            if "antmaze" in config["dataset_name"]:
                last_score_idx = min(20, len(all_scores) - 1)
            if param_3 is None:
                full_scores[config["dataset_name"]][str(config[param_1])][str(config[param_2])].append(
                    all_scores[last_score_idx])
            else:
                full_scores[config["dataset_name"]][str(config[param_1])][str(config[param_2])][str(config[param_3])].append(
                    all_scores[last_score_idx])
            config_list.append(config)
            name_list.append(run.name)
            lasts.append(last)

    return full_scores


def average_seeds(full_scores, is_td3=False, three_params=False):
    S = 0
    full_means = {}
    bests = {}
    for dataset in full_scores:
        ba, bc, bmean, bstd = 0, 0, 0, 0
        for ac in full_scores[dataset]:
            for cc in full_scores[dataset][ac]:
                if not three_params:
                    score = np.mean(full_scores[dataset][ac][cc])
                    std = np.std(full_scores[dataset][ac][cc])
                    if bmean <= score:
                        bmean = score
                        bstd = std
                        ba = ac
                        bc = cc
                    if dataset not in full_means:
                        full_means[dataset] = {}
                    ka = ac
                    if cc not in full_means[dataset]:
                        full_means[dataset][cc] = {}
                    full_means[dataset][cc][ka] = score
                else:
                    for tp in full_scores[dataset][ac][cc]:
                        score = np.mean(full_scores[dataset][ac][cc][tp])
                        std = np.std(full_scores[dataset][ac][cc][tp])
                        if dataset not in full_means:
                            full_means[dataset] = {}
                        ka = ac
                        if cc not in full_means[dataset]:
                            full_means[dataset][cc] = {}
                        if ka not in full_means[dataset][cc]:
                            full_means[dataset][cc][ka] = {}
                        full_means[dataset][cc][ka][tp] = score
        bests[dataset] = {}
        S += bmean
    return full_means


domain2envs = {
    "Gym-MuJoCo": ["hopper", "walker2d", "halfcheetah"],
    "AntMaze": ["antmaze"],
    "Adroit": ["pen", "door", "hammer", "relocate"]
}


def average_domains(full_means, domains_to_proc=["Gym-MuJoCo", "AntMaze", "Adroit"], three_params=False):
    domain_avgereged = {}

    unique_cc = {
        "Gym-MuJoCo": None,
        "AntMaze": None,
        "Adroit": None
    }
    unique_ac = {
        "Gym-MuJoCo": None,
        "AntMaze": None,
        "Adroit": None
    }
    if three_params:
        unique_tp = {
            "Gym-MuJoCo": None,
            "AntMaze": None,
            "Adroit": None
        }

    # print(list(full_means.keys()))
    unique_cc["Gym-MuJoCo"] = list(full_means["hopper-medium-v2"].keys())
    unique_ac["Gym-MuJoCo"] = list(full_means["hopper-medium-v2"][unique_cc["Gym-MuJoCo"][0]].keys())
    if three_params:
        unique_tp["Gym-MuJoCo"] = list(full_means["hopper-medium-v2"][unique_cc["Gym-MuJoCo"][0]][unique_ac["Gym-MuJoCo"][0]].keys())

    unique_cc["AntMaze"] = list(full_means["antmaze-umaze-v2"].keys())
    unique_ac["AntMaze"] = list(full_means["antmaze-umaze-v2"][unique_cc["AntMaze"][0]].keys())
    if three_params:
        unique_tp["AntMaze"] = list(
            full_means["antmaze-umaze-v2"][unique_cc["AntMaze"][0]][unique_ac["AntMaze"][0]].keys())

    unique_cc["Adroit"] = list(full_means["door-expert-v1"].keys())
    unique_ac["Adroit"] = list(full_means["door-expert-v1"][unique_cc["Adroit"][0]].keys())
    if three_params:
        unique_tp["Adroit"] = list(
            full_means["door-expert-v1"][unique_cc["Adroit"][0]][unique_ac["Adroit"][0]].keys())

    for domain in domains_to_proc:
        domain_avgereged[domain] = {}
        for cc in unique_cc[domain]:
            if cc not in domain_avgereged[domain]:
                domain_avgereged[domain][cc] = {}
            for ac in unique_ac[domain]:
                if not three_params:
                    avg = []
                    for data in full_means:
                        is_domain = False
                        for env in domain2envs[domain]:
                            if env in data:
                                is_domain = True
                                break
                        if is_domain:
                            avg.append(full_means[data][cc][ac])
                    domain_avgereged[domain][cc][ac] = np.mean(avg)
                else:
                    if ac not in domain_avgereged[domain][cc]:
                        domain_avgereged[domain][cc][ac] = {}
                    for tp in unique_tp[domain]:
                        avg = []
                        for data in full_means:
                            is_domain = False
                            for env in domain2envs[domain]:
                                if env in data:
                                    is_domain = True
                                    break
                            if is_domain:
                                avg.append(full_means[data][cc][ac][tp])
                        domain_avgereged[domain][cc][ac][tp] = np.mean(avg)

    return domain_avgereged


def listed_avg(data, three_params=False):
    listed_avg = {}

    for env in data:
        if env not in listed_avg:
            listed_avg[env] = []
        for ac in data[env]:
            for cc in data[env][ac]:
                if not three_params:
                    listed_avg[env].append(data[env][ac][cc])
                else:
                    for tp in data[env][ac][cc]:
                        listed_avg[env].append(data[env][ac][cc][tp])
    return listed_avg


def convert_to_lists(full_means, domains_avg, three_params=False):
    listed_all = {}
    listed_domains = {}
    for algo in full_means:
        listed_all[algo] = listed_avg(full_means[algo], three_params=three_params)
        listed_domains[algo] = listed_avg(domains_avg[algo], three_params=three_params)

    return listed_all, listed_domains


def download_data(algo_to_sweeps, param_1="actor_bc_coef", param_2="critic_bc_coef", param_3=None, to_list=True, domains_to_process=["Gym-MuJoCo", "AntMaze", "Adroit"]):
    data = {}
    for algo in algo_to_sweeps:
        print(f"Downloading {algo} data")
        # if "IQL" in algo:
        #     data[algo] = get_data_from_sweeps_iql(algo_to_sweeps[algo])
        # else:    
        data[algo] = get_data_from_sweeps(algo_to_sweeps[algo], param_1, param_2, param_3)

    full_means = {}
    for algo in data:
        full_means[algo] = average_seeds(data[algo], "TD3" in algo, param_3 is not None)

    domains_avg = {}
    for algo in full_means:
        domains_avg[algo] = average_domains(full_means[algo], domains_to_proc=domains_to_process, three_params=param_3 is not None)

    if to_list:
        return convert_to_lists(full_means, domains_avg, three_params=param_3 is not None)
    else:
        return full_means, domains_avg


### Run this code only if you have acceses to the wandb projects
# listed_depth_all_rebrac, listed_depth_domains_rebrac = download_data(
#     {
#         "ReBRAC": ["tarasovd/ReBRAC/sweeps/ge1r6cc4"],
#         "ReBRAC+CE": ["tarasovd/ReBRAC/sweeps/z4m9veg2"]
#     },
#     "critic_n_hiddens",
#     "eval_seed",
#     to_list=False
# )
# with open('bin/depth_rebrac_domains.pickle', 'wb') as handle:
#     pickle.dump(listed_depth_domains_rebrac, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/depth_rebrac_all.pickle', 'wb') as handle:
#     pickle.dump(listed_depth_all_rebrac, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# listed_depth_all_iql, listed_depth_domains_iql = download_data(
#     {
#         "IQL": ["tarasovd/ReBRAC/sweeps/7zvhoxit"],
#         "IQL+CE": ["tarasovd/ReBRAC/sweeps/sqauxbqr"]
#     },
#     "critic_hidden_dims",
#     "eval_seed",
#     to_list=False
# )
#
# with open('bin/depth_iql_domains.pickle', 'wb') as handle:
#     pickle.dump(listed_depth_domains_iql, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/depth_iql_all.pickle', 'wb') as handle:
#     pickle.dump(listed_depth_all_iql, handle, protocol=pickle.HIGHEST_PROTOCOL)

# listed_expand_all_rebrac, listed_expand_domains_rebrac = download_data(
#     {
#         "ReBRAC": ["tarasovd/ReBRAC/sweeps/odqzpd32"],
#     },
#     "v_expand_mode",
#     "v_expand",
#     to_list=False
# )
# with open('bin/expand_rebrac_domains.pickle', 'wb') as handle:
#     pickle.dump(listed_expand_domains_rebrac, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/expand_rebrac_all.pickle', 'wb') as handle:
#     pickle.dump(listed_expand_all_rebrac, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# listed_expand_all_iql, listed_expand_domains_iql = download_data(
#     {
#         "IQL": ["tarasovd/ReBRAC/sweeps/57aliidu"],
#     },
#     "v_expand_mode",
#     "v_expand",
#     to_list=False
# )
#
# with open('bin/expand_iql_domains.pickle', 'wb') as handle:
#     pickle.dump(listed_expand_domains_iql, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/expand_iql_all.pickle', 'wb') as handle:
#     pickle.dump(listed_expand_all_iql, handle, protocol=pickle.HIGHEST_PROTOCOL)


# listed_expand_all_lb_sac, listed_expand_domains_lb_sac = download_data(
#     {
#         "LB-SAC": ["tarasovd/ReBRAC/sweeps/toy22c04"],
#     },
#     "v_expand_mode",
#     "v_expand",
#     to_list=False,
# )
#
# with open('bin/expand_lb_sac_domains.pickle', 'wb') as handle:
#     pickle.dump(listed_expand_domains_lb_sac, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/expand_lb_sac_all.pickle', 'wb') as handle:
#     pickle.dump(listed_expand_all_lb_sac, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
with open('bin/expand_rebrac_domains.pickle', 'rb') as handle:
    listed_expand_domains_rebrac = pickle.load(handle)
with open('bin/expand_rebrac_all.pickle', 'rb') as handle:
    listed_expand_all_rebrac = pickle.load(handle)
with open('bin/expand_iql_domains.pickle', 'rb') as handle:
    listed_expand_domains_iql = pickle.load(handle)
with open('bin/expand_iql_all.pickle', 'rb') as handle:
    listed_expand_all_iql = pickle.load(handle)

with open('bin/expand_lb_sac_domains.pickle', 'rb') as handle:
    listed_expand_domains_lb_sac = pickle.load(handle)
with open('bin/expand_lb_sac_all.pickle', 'rb') as handle:
    listed_expand_all_lb_sac = pickle.load(handle)

with open('bin/depth_rebrac_domains.pickle', 'rb') as handle:
    listed_depth_domains_rebrac = pickle.load(handle)
with open('bin/depth_rebrac_all.pickle', 'rb') as handle:
    listed_depth_all_rebrac = pickle.load(handle)
with open('bin/depth_iql_domains.pickle', 'rb') as handle:
    listed_depth_domains_iql = pickle.load(handle)
with open('bin/depth_iql_all.pickle', 'rb') as handle:
    listed_depth_all_iql = pickle.load(handle)


def plot_expand_domain(data, domain, color, type):
    for algo in data:
        marker = color
        if "min" in type:
            marker += "--"
        else:
            marker += "-"
        plt.plot([-0.05, 0.0, 0.05, 0.1, 0.2], data[algo][domain], marker, label=algo+f', {type}')


def transpose_dict(d):
    transposed = {}

    for sub_key in next(iter(d.values())):
        transposed[sub_key] = {}

    for main_key, sub_dict in d.items():
        for sub_key, value in sub_dict.items():
            transposed[sub_key][main_key] = value

    return transposed

def proc_expand(data):
    res_min = {}
    res_both = {}
    for algo in data:
        res_min[algo] = {}
        res_both[algo] = {}
        for domain in data[algo]:
            res_min[algo][domain] = {}
            res_both[algo][domain] = {}
            data[algo][domain] = transpose_dict(data[algo][domain])
            for percent in data[algo][domain]['min']:
                res_min[algo][domain][int(float(percent) * 100)] = data[algo][domain]['min'][percent]
            for percent in data[algo][domain]['both']:
                res_both[algo][domain][int(float(percent) * 100)] = data[algo][domain]['both'][percent]
            res_min[algo][domain] = [res_min[algo][domain][d] for d in sorted(list(res_min[algo][domain].keys()))]
            res_both[algo][domain] = [res_both[algo][domain][d] for d in sorted(list(res_both[algo][domain].keys()))]
            avg_zero = (res_min[algo][domain][1] + res_both[algo][domain][1]) / 2
            res_min[algo][domain][1] = avg_zero
            res_both[algo][domain][1] = avg_zero
    return res_min, res_both


iql_expand_min, iql_expand_both = proc_expand(listed_expand_domains_iql)
lb_sac_expand_min, lb_sac_expand_both = proc_expand(listed_expand_domains_lb_sac)
rebrac_expand_min, rebrac_expand_both = proc_expand(listed_expand_domains_rebrac)


for domain in ["Gym-MuJoCo", "AntMaze", "Adroit"]:
    plot_expand_domain(rebrac_expand_min, domain, "ro", "min")
    plot_expand_domain(rebrac_expand_both, domain, "ro", "both")
    plot_expand_domain(iql_expand_min, domain, "go", "min")
    plot_expand_domain(iql_expand_both, domain, "go", "both")
    # if domain == "Gym-MuJoCo":
    plot_expand_domain(lb_sac_expand_min, domain, "bo", "min")
    plot_expand_domain(lb_sac_expand_both, domain, "bo", "both")
    plt.grid()
    plt.legend()
    plt.xlabel('Expansion size')
    plt.ylabel('Average score')
    plt.title(domain)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xticks([-0.05, 0.0, 0.05, 0.1, 0.2])
    # if save_name:
    plt.savefig(f"out/expand_{domain}.pdf", bbox_inches='tight', dpi=300)
    plt.close()


print("Expand ReBRAC both")
print(rebrac_expand_both)
print("Expand ReBRAC min")
print(rebrac_expand_min)
print("Expand IQL both")
print(iql_expand_both)
print("Expand IQL min")
print(iql_expand_min)
print("Expand LB-SAC both")
print(lb_sac_expand_both)
print("Expand LB-SAC min")
print(lb_sac_expand_min)

# raise ValueError()

def plot_depth_domain(data, domain, color):
    for algo in data:
        marker = color
        if "CE" in algo:
            marker += "--"
        else:
            marker += "-"
        plt.plot(np.arange(0, len(data[algo][domain])), data[algo][domain], marker, label=algo)


def proc_iql_depth(data):
    res = {}
    for algo in data:
        res[algo] = {}
        for domain in data[algo]:
            res[algo][domain] = {}

            for depth in data[algo][domain]['42']:
                res[algo][domain][depth.count('256') - 2] = data[algo][domain]['42'][depth]
            res[algo][domain] = [res[algo][domain][d] for d in sorted(list(res[algo][domain].keys()))]
    return res


def proc_rebrac_depth(data):
    res = {}
    for algo in data:
        res[algo] = {}
        for domain in data[algo]:
            res[algo][domain] = {}

            for depth in data[algo][domain]['42']:
                res[algo][domain][int(depth) - 3] = data[algo][domain]['42'][depth]
            res[algo][domain] = [res[algo][domain][d] for d in sorted(list(res[algo][domain].keys()))]
    return res


iql_processed = proc_iql_depth(listed_depth_domains_iql)
rebrac_processed = proc_rebrac_depth(listed_depth_domains_rebrac)

for domain in ["Gym-MuJoCo", "AntMaze", "Adroit"]:
    plot_depth_domain(rebrac_processed, domain, "ro")
    plot_depth_domain(iql_processed, domain, "go")
    plt.grid()
    plt.legend()
    plt.xlabel('Number of additional critic layers')
    plt.ylabel('Average score')
    plt.title(domain)
    plt.legend(loc='center left', bbox_to_anchor=(1, 0.5))

    plt.xticks(list(np.arange(0, 4 + 1, 1)))  # + [12])
    # if save_name:
    plt.savefig(f"out/depth_{domain}.pdf", bbox_inches='tight', dpi=300)
    plt.close()


### Run this code only if you have acceses to the wandb projects
# listed_all_rebrac_at, listed_domains_rebrac_at = download_data(
#     {
#         "ReBRAC+CE+AT": ["tarasovd/ReBRAC/sweeps/fmfsyi2p", "tarasovd/ReBRAC/sweeps/qb7t67el", "tarasovd/ReBRAC/sweeps/w5pnuxc0"],
# }
# )
#
# with open('bin/eop_rebrac_at_domains.pickle', 'wb') as handle:
#     pickle.dump(listed_domains_rebrac_at, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/eop_rebrac_at_all.pickle', 'wb') as handle:
#     pickle.dump(listed_all_rebrac_at, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# all_rebrac_ct, domains_rebrac_ct = download_data(
#      {
#          "ReBRAC+CE+CT": ["tarasovd/ReBRAC/sweeps/rsi478m7"],
#      },
#     "n_classes",
#     "sigma_frac",
#     to_list=False,
# )
#
# with open('bin/rebrac_ct_domains.pickle', 'wb') as handle:
#     pickle.dump(domains_rebrac_ct, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/rebrac_ct_all.pickle', 'wb') as handle:
#     pickle.dump(all_rebrac_ct, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# listed_all_iql, listed_domains_iql = download_data(
#     {
#         "IQL": ["tarasovd/ReBRAC/sweeps/phaq0abh"],
#         "IQL+CE+AT": ["tarasovd/ReBRAC/sweeps/2nzcdijr"],
#     },
#     "temperature",
#     "expectile",
# )
#
# with open('bin/eop_iql_domains.pickle', 'wb') as handle:
#     pickle.dump(listed_domains_iql, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/eop_iql_all.pickle', 'wb') as handle:
#     pickle.dump(listed_all_iql, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# all_iql_ct, domains_iql_ct = download_data(
#      {
#          "IQL+CE+CT": ["tarasovd/ReBRAC/sweeps/u5dp3rg5"],
#      },
#     "n_classes",
#     "sigma_frac",
#     to_list=False,
# )
#
# with open('bin/iql_ct_domains.pickle', 'wb') as handle:
#     pickle.dump(domains_iql_ct, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/iql_ct_all.pickle', 'wb') as handle:
#     pickle.dump(all_iql_ct, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
#
# listed_all_lb_sac, listed_domains_lb_sac = download_data(
#     {
#         "LB-SAC": ["tarasovd/ReBRAC/sweeps/jc8wpm3x"],
#         "LB-SAC+CE+AT": ["tarasovd/ReBRAC/sweeps/j6pegfn2"],
#     },
#     "num_critics",
#     "eval_seed",
# )
#
# with open('bin/eop_lb_sac_domains.pickle', 'wb') as handle:
#     pickle.dump(listed_domains_lb_sac, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/eop_lb_sac_all.pickle', 'wb') as handle:
#     pickle.dump(listed_all_lb_sac, handle, protocol=pickle.HIGHEST_PROTOCOL)

# all_lb_sac_ct, domains_lb_sac_ct = download_data(
#      {
#          "LB-SAC+CE+CT": ["tarasovd/ReBRAC/sweeps/ohwunc39"],
#      },
#     "n_classes",
#     "sigma_frac",
#     to_list=False,
# )
#
# with open('bin/domain_lb_sac_ct.pickle', 'wb') as handle:
#     pickle.dump(domains_lb_sac_ct, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/all_lb_sac_hp_ct.pickle', 'wb') as handle:
#     pickle.dump(all_lb_sac_ct, handle, protocol=pickle.HIGHEST_PROTOCOL)


# all_rebrac_full, domains_rebrac_full = download_data(
#      {
#          "ReBRAC+CE+FULL": ["tarasovd/ReBRAC/sweeps/4zvx44nm"],
#      },
#     "n_classes",
#     "actor_bc_coef",
#     "v_expand",
# )
#
# with open('bin/rebrac_full_domains.pickle', 'wb') as handle:
#     pickle.dump(domains_rebrac_full, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/rebrac_full_all.pickle', 'wb') as handle:
#     pickle.dump(all_rebrac_full, handle, protocol=pickle.HIGHEST_PROTOCOL)
#
# all_iql_full, domains_iql_full = download_data(
#      {
#          "IQL+CE+FULL": ["tarasovd/ReBRAC/sweeps/7sbwhnxh"],
#      },
#     "n_classes",
#     "expectile",
#     "v_expand",
# )

# with open('bin/iql_full_domains.pickle', 'wb') as handle:
#     pickle.dump(domains_iql_full, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/iql_full_all.pickle', 'wb') as handle:
#     pickle.dump(all_iql_full, handle, protocol=pickle.HIGHEST_PROTOCOL)


# all_lb_sac_full, domains_lb_sac_full = download_data(
#      {
#          "LB-SAC+CE+FULL": ["tarasovd/ReBRAC/sweeps/djuujo4o"],
#      },
#     "n_classes",
#     "num_critics",
#     "v_expand",
# )
#
# with open('bin/lb_sac_full_domains.pickle', 'wb') as handle:
#     pickle.dump(domains_lb_sac_full, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/lb_sac_full_all.pickle', 'wb') as handle:
#     pickle.dump(all_lb_sac_full, handle, protocol=pickle.HIGHEST_PROTOCOL)



with open('bin/rebrac_full_domains.pickle', 'rb') as handle:
    listed_domains_rebrac_full = pickle.load(handle)
with open('bin/rebrac_full_all.pickle', 'rb') as handle:
    listed_all_rebrac_full = pickle.load(handle)
# listed_all_rebrac_full, listed_domains_rebrac_full = convert_to_lists(listed_all_rebrac_full, listed_domains_rebrac_full, three_params=True)

with open('bin/iql_full_domains.pickle', 'rb') as handle:
    listed_domains_iql_full = pickle.load(handle)
with open('bin/iql_full_all.pickle', 'rb') as handle:
    listed_all_iql_full = pickle.load(handle)

with open('bin/lb_sac_full_domains.pickle', 'rb') as handle:
    listed_domains_lb_sac_full = pickle.load(handle)
with open('bin/lb_sac_full_all.pickle', 'rb') as handle:
    listed_all_lb_sac_full = pickle.load(handle)

# listed_all_iql_full, listed_domains_iql_full = convert_to_lists(listed_all_iql_full, listed_domains_iql_full, three_params=True)
# raise ValueError()

# with open('bin/eop_rebrac_at_domains.pickle', 'rb') as handle:
#     listed_domains_rebrac = pickle.load(handle)
# with open('bin/eop_rebrac_at_all.pickle', 'rb') as handle:
#     listed_all_rebrac = pickle.load(handle)

with open('bin/rebrac_ct_domains.pickle', 'rb') as handle:
    domains_rebrac_ct = pickle.load(handle)
with open('bin/rebrac_ct_all.pickle', 'rb') as handle:
    all_rebrac_ct = pickle.load(handle)

# with open('bin/eop_iql_domains.pickle', 'rb') as handle:
#     listed_domains_iql = pickle.load(handle)
# with open('bin/eop_iql_all.pickle', 'rb') as handle:
#     listed_all_iql = pickle.load(handle)

with open('bin/iql_ct_domains.pickle', 'rb') as handle:
    domains_iql_ct = pickle.load(handle)
with open('bin/iql_ct_all.pickle', 'rb') as handle:
    all_iql_ct = pickle.load(handle)

# with open('bin/eop_lb_sac_domains.pickle', 'rb') as handle:
#     listed_domains_lb_sac = pickle.load(handle)
# with open('bin/eop_lb_sac_all.pickle', 'rb') as handle:
#     listed_all_lb_sac = pickle.load(handle)

with open('bin/domain_lb_sac_ct.pickle', 'rb') as handle:
    domains_lb_sac_ct = pickle.load(handle)
with open('bin/all_lb_sac_hp_ct.pickle', 'rb') as handle:
    all_lb_sac_ct = pickle.load(handle)


listed_all_rebrac_ct, listed_domains_rebrac_ct = convert_to_lists(all_rebrac_ct, domains_rebrac_ct)
listed_all_iql_ct, listed_domains_iql_ct = convert_to_lists(all_iql_ct, domains_iql_ct)
listed_all_lb_sac_ct, listed_domains_lb_sac_ct = convert_to_lists(all_lb_sac_ct, domains_lb_sac_ct)

def plot_cls_params_heatmap(data, title, algo, save_name=None):
    parameters = sorted(map(lambda x: float(x), data.keys()), reverse=True)
    values = sorted(map(lambda x: int(x), data[list(data.keys())[0]].keys()))

    if title == "Locomotion":
        title = "Gym-MuJoCo"
    # Create a matrix of scores
    scores_matrix = np.array([[data[str(parameter)][str(value)] for value in values] for parameter in parameters])
    # print(scores_matrix)
    # print("Average for m")
    avg_m = np.mean(scores_matrix, axis=0)
    # print("Average for sigma/zeta")
    avg_sigma = np.mean(scores_matrix, axis=1)
    # Plot the heatmap
    plt.figure(figsize=(10, 6))
    plt.imshow(scores_matrix, cmap='viridis', interpolation='nearest')

    # Set ticks and labels
    plt.xticks(np.arange(len(values)), values)
    plt.yticks(np.arange(len(parameters)), parameters)
    plt.xlabel('Number of classes')
    plt.ylabel('$\sigma / \zeta$')
    plt.title(algo + " " + title)

    # Add color bar
    plt.colorbar(label='Scores')

    # Add cell values
    for i in range(len(parameters)):
        for j in range(len(values)):
            plt.text(j, i, f'{scores_matrix[i, j]:.2f}', ha='center', va='center',
                     color='white' if scores_matrix[i, j] > np.max(scores_matrix) / 2 else 'black')
    if save_name:
        plt.savefig(save_name, bbox_inches='tight', dpi=300)
    plt.close()
    return avg_m, avg_sigma


def plot_heatmaps(data, algo):
    avg_m, avg_sigma = {}, {}
    for k in data:
        (m, s) = plot_cls_params_heatmap(data[k], k, algo, save_name=f'out/{algo}_{k}_hp.pdf')
        avg_m[k] = m
        avg_sigma[k] = s[::-1]
    return avg_m, avg_sigma


# In[109]:


def convert_impact_list_to_tex(impact, algo):
    algo = "{" + algo + "}"
    return "& \\textbf{" + algo + "} & " + " & ".join(map(str, impact)) + " \\\\"


plot_heatmaps(all_lb_sac_ct['LB-SAC+CE+CT'], 'LB-SAC')
lb_sac_m, lb_sac_s = plot_heatmaps(domains_lb_sac_ct['LB-SAC+CE+CT'], 'LB-SAC')


print("classes impact")
for k in lb_sac_m:
    print(k)
    print(convert_impact_list_to_tex(lb_sac_m[k], "LB-SAC+CE"))
print()
print("sigma impact")
for k in lb_sac_s:
    print(k)
    print(convert_impact_list_to_tex(lb_sac_s[k], "LB-SAC+CE"))


plot_heatmaps(all_iql_ct['IQL+CE+CT'], 'IQL')
iql_m, iql_s = plot_heatmaps(domains_iql_ct['IQL+CE+CT'], 'IQL')


print("classes impact")
for k in iql_m:
    print(k)
    print(convert_impact_list_to_tex(iql_m[k], "IQL"))
print()
print("sigma impact")
for k in iql_s:
    print(k)
    print(convert_impact_list_to_tex(iql_s[k], "IQL"))

plot_heatmaps(all_rebrac_ct['ReBRAC+CE+CT'], "ReBRAC")
rebrac_m, rebrac_s = plot_heatmaps(domains_rebrac_ct['ReBRAC+CE+CT'], "ReBRAC")


print("classes impact")
for k in rebrac_m:
    print(k)
    print(convert_impact_list_to_tex(rebrac_m[k], "ReBRAC"))
print()
print("sigma impact")
for k in rebrac_s:
    print(k)
    print(convert_impact_list_to_tex(rebrac_s[k], "ReBRAC"))


# Files from ReBRAC https://github.com/DT6A/ReBRAC/tree/public-release/eop/bin
# with open('bin/eop_domains.pickle', 'rb') as handle:
#     listed_domains = pickle.load(handle)
#     del listed_domains["IQL"]
# with open('bin/eop_all.pickle', 'rb') as handle:
#     listed_all = pickle.load(handle)
#     del listed_all["IQL"]
#
# with open('bin/eop_rebrac_at_domains.pickle', 'rb') as handle:
#     listed_domains.update(**pickle.load(handle))
# with open('bin/eop_rebrac_at_all.pickle', 'rb') as handle:
#     listed_all.update(**pickle.load(handle))
# with open('bin/eop_iql_domains.pickle', 'rb') as handle:
#     listed_domains.update(**pickle.load(handle))
# with open('bin/eop_iql_all.pickle', 'rb') as handle:
#     listed_all.update(**pickle.load(handle))
# with open('bin/eop_lb_sac_domains.pickle', 'rb') as handle:
#     listed_domains.update(**pickle.load(handle))
# with open('bin/eop_lb_sac_all.pickle', 'rb') as handle:
#     listed_all.update(**pickle.load(handle))
#
# for k in list(listed_domains.keys()):
#     if "ReBRAC" not in k and "IQL" not in k and "LB-SAC" not in k:
#         del listed_domains[k]
#         del listed_all[k]
#
#
# listed_domains.update(**listed_domains_rebrac_ct)
# listed_all.update(**listed_all_rebrac_ct)
# listed_domains.update(**listed_domains_iql_ct)
# listed_all.update(**listed_all_iql_ct)
# listed_domains.update(**listed_domains_lb_sac_ct)
# listed_all.update(**listed_all_lb_sac_ct)
#
#
# with open('bin/eop_cls_domains.pickle', 'wb') as handle:
#     pickle.dump(listed_domains, handle, protocol=pickle.HIGHEST_PROTOCOL)
# with open('bin/eop_cls_all.pickle', 'wb') as handle:
#     pickle.dump(listed_all, handle, protocol=pickle.HIGHEST_PROTOCOL)

with open('bin/eop_cls_domains.pickle', 'rb') as handle:
    listed_domains = pickle.load(handle)
with open('bin/eop_cls_all.pickle', 'rb') as handle:
    listed_all = pickle.load(handle)

listed_domains.update(**listed_domains_rebrac_full)
listed_domains.update(**listed_domains_iql_full)
listed_domains.update(**listed_domains_lb_sac_full)

def print_tables(data, algorithms, points=[0, 1, 2, 4, 9, 14, 19]):
    # algorithms = ["TD3 + BC", "IQL", "ReBRAC"]
    print(algorithms)
    fst_algo = algorithms[0]

    all_keys = list(sorted(data[fst_algo].keys()))
    all_values = {
        algo: [data[algo][n] if n in data[algo] else data[algo]["Gym-MuJoCo"] for n in all_keys] for algo in algorithms
    }
    for i, name in enumerate(all_keys):
        print("=" * 30)
        print(name)
        print()
        rewards = [data[algo][name] if name in data[algo] else data[algo]["Gym-MuJoCo"] for algo in algorithms]
        max_runs = max(map(len, rewards))
        x = np.arange(max_runs) + 1
        for algo, reward in zip(algorithms, rewards):
            perf = expected_online_performance(reward, len(reward))
            means = np.array(perf['mean'])
            stds = np.array(perf['std'])
            print("& \\textbf{" + algo + "} &", end=" ")
            for point in points:
                if point >= len(reward):
                    print("-", end=(" & " if point != 19 else "\\\\\n"))
                else:
                    print("{:3.1f}".format(means[point]), "$\pm$", "{:3.1f}".format(stds[point]),
                          end=(" & " if point != 19 else "\\\\\n"))


def print_v_tables(data, algorithms):
    # algorithms = ["TD3 + BC", "IQL", "ReBRAC"]
    print(algorithms)
    fst_algo = algorithms[0]

    locomotion_envs = ["halfcheetah", "hopper", "walker2d"]
    adroit_envs = ["pen", "door", "hammer", "relocate"]

    locomotion_datasets = [
        "random-v2",
        "medium-v2",
        "expert-v2",
        "medium-expert-v2",
        "medium-replay-v2",
        "full-replay-v2",
    ]
    antmaze_datasets = [
        "umaze-v2",
        "medium-play-v2",
        "large-play-v2",
        "umaze-diverse-v2",
        "medium-diverse-v2",
        "large-diverse-v2",
    ]
    adroit_datasets = [
        "human-v1",
        "cloned-v1",
        "expert-v1",
    ]

    concated = {
        env: {} for env in locomotion_envs + ["antmaze"] + adroit_envs
    }

    for env in locomotion_envs:
        for dataset in locomotion_datasets:
            concated[env][dataset] = ["" for _ in range(20)]
    for env in ["antmaze"]:
        for dataset in antmaze_datasets:
            concated[env][dataset] = ["" for _ in range(20)]
    for env in adroit_envs:
        for dataset in adroit_datasets:
            concated[env][dataset] = ["" for _ in range(20)]

    all_keys = list(sorted(data[fst_algo].keys()))
    all_values = {
        algo: [data[algo][n] if n in data[algo] else data[algo]["Gym-MuJoCo"] for n in all_keys] for algo in algorithms
    }
    for i, name in enumerate(all_keys):
        env_name = name.split('-')[0]
        dataset_name = '-'.join(name.split('-')[1:])

        rewards = [data[algo][name] for algo in algorithms]
        max_runs = max(map(len, rewards))
        x = np.arange(max_runs) + 1
        for point in range(20):
            alg_n = 0
            max_idx = 0
            max_val = -10
            strings = []
            for algo, reward in zip(algorithms, rewards):
                perf = expected_online_performance(reward, len(reward))
                means = np.array(perf['mean'])
                stds = np.array(perf['std'])
                if point >= len(means):
                    strings.append("-")
                else:
                    strings.append("{:3.1f}".format(means[point]) + " $\\pm$ " + "{:3.1f}".format(stds[point]))
                    if max_val < means[point]:
                        max_val = means[point]
                        max_idx = alg_n
                alg_n += 1
            strings[max_idx] = "\\textbf{" + strings[max_idx] + "}"
            concated[env_name][dataset_name][point] = " & ".join(strings) + " & "

    for envs, datasets in zip([locomotion_envs, ["antmaze"], adroit_envs],
                              [locomotion_datasets, antmaze_datasets, adroit_datasets]):
        for env in envs:
            print("=" * 30)
            print(env)
            for i in range(20):
                print(i + 1, "&", ("".join([concated[env][dataset][i] for dataset in datasets]))[:-2], "\\\\")

            # In[57]:


print("EOP:")
print_tables(listed_domains,
             ["ReBRAC", "ReBRAC+CE+AT", "ReBRAC+CE+CT", "IQL", "IQL+CE+AT", "IQL+CE+CT", "LB-SAC", "LB-SAC+CE+AT"])

print_tables(listed_domains, ["LB-SAC+CE+CT"], points=[0, 1, 2, 4, 8, 14, 19])

# print(listed_domains["ReBRAC"])
# print(listed_domains["ReBRAC+CE+FULL"])
print_tables(listed_domains,
             ["ReBRAC", "ReBRAC+CE+FULL", "IQL", "IQL+CE+FULL", "LB-SAC", "LB-SAC+CE+FULL"], points=[0, 1, 2, 4, 8, 14, 17])