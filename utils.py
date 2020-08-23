import random

import colorama as color
import numpy as np
import yaml

import plotter


def read_config(verbose=False):
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if verbose:
        print(f"CONF: {data}")
    return data


def print_blue(message):
    print(f"{color.Fore.BLUE}{message}{color.Style.RESET_ALL}")


def get_mean_costs(raw_stats):
    cps = np.array([d['costs_per_timeslot'] for d in raw_stats])
    return {
        "mean": cps.mean(axis=0),
        "var": cps.var(axis=0)
    }


def get_mean_processed_jobs(raw_stats):
    pj = np.array([d['processed_jobs_per_timeslot'] for d in raw_stats])
    return {
        "mean": pj.mean(axis=0),
        "var": pj.var(axis=0)
    }


def get_mean_lost_jobs(raw_stats):
    lj = np.array([d['lost_jobs_per_timeslot'] for d in raw_stats])
    return {
        "mean": lj.mean(axis=0),
        "var": lj.var(axis=0)
    }


def get_mean_wait_time(raw_stats):
    raw_wt = [d['wait_time_per_job'] for d in raw_stats]

    max_len = max(len(x) for x in raw_wt)
    wt = []
    for simulation in raw_wt:
        wt.append(simulation + [0] * (max_len - len(simulation)))
    wt = np.array(wt)
    return {
        "mean": wt.mean(axis=0),
        "var": wt.var(axis=0)
    }


# 3 azioni (0,1,2) e 6 stati
def generate_random_policy(states_num, action_num):
    rpolicy = []
    for s in range(states_num):
        rpolicy.append(random.randint(0, action_num - 1))
    return tuple(rpolicy)


def easy_plot(projectname, comment, stats, view):
    plotter.plot_cumulative({"costs": stats['costs_per_timeslot'],
                             "processed jobs": stats['processed_jobs_per_timeslot'],
                             "lost jobs": stats['lost_jobs_per_timeslot']},
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative ({comment})",
                            projectname=projectname, view=view)

    plotter.plot({"costs per ts": stats['costs_per_timeslot'],
                  "processed jobs per ts": stats['processed_jobs_per_timeslot'],
                  "lost jobs per ts": stats['lost_jobs_per_timeslot']},
                 xlabel="timeslot", title=f"[{projectname}] Mean per Timeslot ({comment})",
                 projectname=projectname, view=view)

    # total time is wait time in the system
    plotter.bar({"job wait time": stats['wait_time_per_job']},
                xlabel="timeslot", ylabel="percentage of wait time",
                title=f"[{projectname}] Mean Job Total Time ({comment})",
                projectname=projectname, view=view)
