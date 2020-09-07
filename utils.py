import random
import logging
import subprocess

import colorama as color
import numpy as np
import yaml

import plotter
from state import State


def get_last_commit_link():
    commit_hash = subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
    return f"https://github.com/AlessandroSpallina/Slicing-5G-MDP/commit/{commit_hash[:-1].decode('utf-8')}"


def read_config(verbose=False):
    with open('config.yaml') as f:
        data = yaml.load(f, Loader=yaml.FullLoader)
    if verbose:
        logging.info(f"CONF: {data}")
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


def get_mean_jobs_in_queue(raw_stats):
    raw_jiq = [d['state_sequence'] for d in raw_stats]

    jiq = np.array([[d.k for d in simulation] for simulation in raw_jiq])

    return {
        "mean": jiq.mean(axis=0),
        "var": jiq.var(axis=0)
    }


def get_mean_active_servers(raw_stats):
    raw_as = [d['state_sequence'] for d in raw_stats]

    asr = np.array([[d.n for d in simulation] for simulation in raw_as])

    return {
        "mean": asr.mean(axis=0),
        "var": asr.var(axis=0)
    }


# 3 azioni (0,1,2) e 6 stati
def generate_random_policy(states_num, action_num):
    rpolicy = []
    for s in range(states_num):
        rpolicy.append(random.randint(0, action_num - 1))
    return tuple(rpolicy)


def easy_plot(projectname, stats, comment="", view=False):
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

    plotter.plot_two_scales(stats['jobs_in_queue_per_timeslot'], stats['active_servers_per_timeslot'],
                            ylabel1="jobs in queue", ylabel2="active servers", xlabel="timeslot",
                            title=f"[{projectname}] Mean Queue and Servers", projectname=projectname, view=view)


def comparison_plot(projectname, comparison_stats, comment="", view=False):
    plotter.plot_cumulative({"mdp": comparison_stats['mdp']['costs_per_timeslot'],
                             "random": comparison_stats['random']['costs_per_timeslot']},
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative Costs ({comment})",
                            projectname=projectname, view=view)

    plotter.plot_cumulative({"mdp": comparison_stats['mdp']['processed_jobs_per_timeslot'],
                             "random": comparison_stats['random']['processed_jobs_per_timeslot']}, ylabel="job",
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative Processed Jobs ({comment})",
                            projectname=projectname, view=view)

    plotter.plot_cumulative({"mdp": comparison_stats['mdp']['lost_jobs_per_timeslot'],
                             "random": comparison_stats['random']['lost_jobs_per_timeslot']}, ylabel="job",
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative Lost Jobs ({comment})",
                            projectname=projectname, view=view)

    plotter.scatter({"mdp": comparison_stats['mdp']['wait_time_per_job'],
                     "random": comparison_stats['random']['wait_time_per_job']}, xlabel="timeslot",
                    ylabel="percentage of wait time", title=f"[{projectname}] Mean Job Total Time ({comment})",
                    projectname=projectname, view=view)
