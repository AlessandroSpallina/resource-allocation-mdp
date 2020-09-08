import random
import logging
import subprocess

import colorama as color
import numpy as np
import yaml

import plotter


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


def get_matrix_policy(policy, server_max_cap):
    return np.array(np.split(np.array(policy), server_max_cap + 1)).transpose().tolist()


# 3 azioni (0,1,2) e 6 stati
def generate_random_policy(states_num, action_num):
    rpolicy = []
    for s in range(states_num):
        rpolicy.append(random.randint(0, action_num - 1))
    return tuple(rpolicy)


def average_plot_points(data, average_window):
    chunks = np.split(data, average_window)
    averaged = [chunk.mean() for chunk in chunks]

    return np.arange(len(data)/average_window, step=average_window), averaged


def easy_plot(projectname, stats, max_points_in_plot, view=False):

    cost_per_ts = average_plot_points(stats['costs_per_timeslot'], max_points_in_plot)
    processed_per_ts = average_plot_points(stats['processed_jobs_per_timeslot'], max_points_in_plot)
    lost_per_ts = average_plot_points(stats['lost_jobs_per_timeslot'], max_points_in_plot)
    jobs_in_queue_per_ts = average_plot_points(stats['lost_jobs_per_timeslot'], max_points_in_plot)
    active_server_per_ts = average_plot_points(stats['active_servers_per_timeslot'], max_points_in_plot)

    plotter.table([f'{i} jobs' for i in range(len(stats['policy']))],
                  [f'{i} servers' for i in range(len(stats['policy'][0]))],
                  stats['policy'], title=f"[{projectname}] Policy", projectname=projectname, view=view)

    plotter.plot_cumulative(ydata={"costs": cost_per_ts[1],
                                   "processed jobs": processed_per_ts[1],
                                   "lost jobs": lost_per_ts[1]},
                            xdata=cost_per_ts[0], xlabel="timeslot", title=f"[{projectname}] Mean Cumulative",
                            projectname=projectname, view=view)

    plotter.plot(ydata={"costs per ts": cost_per_ts[1],
                        "processed jobs per ts": processed_per_ts[1],
                        "lost jobs per ts": lost_per_ts[1]},
                 xdata=cost_per_ts[0], xlabel="timeslot", title=f"[{projectname}] Mean per Timeslot",
                 projectname=projectname, view=view)

    # total time is wait time in the system
    plotter.bar(ydata={"job wait time": stats['wait_time_per_job']},
                xlabel="timeslot", ylabel="percentage of wait time",
                title=f"[{projectname}] Mean Job Total Time (wait time in the system)",
                projectname=projectname, view=view)

    plotter.plot_two_scales(jobs_in_queue_per_ts[1], active_server_per_ts[1], xdata=jobs_in_queue_per_ts[0],
                            ylabel1="jobs in queue", ylabel2="active servers", xlabel="timeslot",
                            title=f"[{projectname}] Mean Queue and Servers", projectname=projectname, view=view)


def comparison_plot(projectname, comparison_stats, max_points_in_plot, view=False):

    mdp_cost_per_ts = average_plot_points(comparison_stats['mdp']['costs_per_timeslot'], max_points_in_plot)
    mdp_processed_per_ts = average_plot_points(comparison_stats['mdp']['processed_jobs_per_timeslot'], max_points_in_plot)
    mdp_lost_per_ts = average_plot_points(comparison_stats['mdp']['lost_jobs_per_timeslot'], max_points_in_plot)
    mdp_jobs_in_queue_per_ts = average_plot_points(comparison_stats['mdp']['lost_jobs_per_timeslot'], max_points_in_plot)
    mdp_active_server_per_ts = average_plot_points(comparison_stats['mdp']['active_servers_per_timeslot'], max_points_in_plot)

    random_cost_per_ts = average_plot_points(comparison_stats['random']['costs_per_timeslot'], max_points_in_plot)
    random_processed_per_ts = average_plot_points(comparison_stats['random']['processed_jobs_per_timeslot'], max_points_in_plot)
    random_lost_per_ts = average_plot_points(comparison_stats['random']['lost_jobs_per_timeslot'], max_points_in_plot)
    random_jobs_in_queue_per_ts = average_plot_points(comparison_stats['random']['lost_jobs_per_timeslot'], max_points_in_plot)
    random_active_server_per_ts = average_plot_points(comparison_stats['random']['active_servers_per_timeslot'], max_points_in_plot)

    plotter.plot_cumulative(ydata={"mdp": mdp_cost_per_ts[1],
                                   "random": random_cost_per_ts[1]}, xdata=mdp_cost_per_ts[0],
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative Costs",
                            projectname=projectname, view=view)

    plotter.plot_cumulative({"mdp": mdp_processed_per_ts[1],
                             "random": random_processed_per_ts[1]}, ylabel="job", xdata=mdp_processed_per_ts[0],
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative Processed Jobs",
                            projectname=projectname, view=view)

    plotter.plot_cumulative({"mdp": mdp_lost_per_ts[1],
                             "random": random_lost_per_ts[1]}, ylabel="job", xdata=mdp_lost_per_ts[0],
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative Lost Jobs",
                            projectname=projectname, view=view)

    plotter.plot(ydata={"mdp": comparison_stats['mdp']['wait_time_per_job'],
                        "random": comparison_stats['random']['wait_time_per_job']}, xlabel="timeslot",
                 ylabel="percentage of wait time", title=f"[{projectname}] Mean Job Total Time (wait time in the system)",
                 projectname=projectname, view=view)

    plotter.plot(ydata={"mdp": mdp_jobs_in_queue_per_ts[1], "random": random_jobs_in_queue_per_ts[1]},
                 xdata=mdp_jobs_in_queue_per_ts[0], xlabel="timeslot",
                 title=f"[{projectname}] Jobs in queue per Timeslot", projectname=projectname, view=view)

    plotter.plot(ydata={"mdp": mdp_active_server_per_ts[1], "random": random_active_server_per_ts[1]},
                 xdata=mdp_active_server_per_ts[0], xlabel="timeslot",
                 title=f"[{projectname}] Active server per Timeslot", projectname=projectname, view=view)
