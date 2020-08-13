import colorama as color
import plotter
import numpy as np
import random


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


def easy_plot(projectname, stats, view):
    mean_costs = get_mean_costs(stats)
    mean_processed_jobs = get_mean_processed_jobs(stats)
    mean_lost_jobs = get_mean_lost_jobs(stats)
    mean_wait_time = get_mean_wait_time(stats)

    # plots of one simulation
    plotter.plot_cumulative({"costs": stats[0]['costs_per_timeslot'],
                             "processed jobs": stats[0]['processed_jobs_per_timeslot'],
                             "lost jobs": stats[0]['lost_jobs_per_timeslot']},
                            xlabel="timeslot", title=f"[{projectname}] One Sim Cumulative",
                            projectname=projectname, view=view)
    plotter.plot({"costs": stats[0]['costs_per_timeslot'],
                  "processed jobs": stats[0]['processed_jobs_per_timeslot'],
                  "lost jobs": stats[0]['lost_jobs_per_timeslot']},
                 xlabel="timeslot", title=f"[{projectname}] One Sim per Timeslot",
                 projectname=projectname, view=view)

    # plotting of the mean of N simulations
    plotter.plot_cumulative({"costs": mean_costs['mean'],
                             "processed jobs": mean_processed_jobs['mean'],
                             "lost jobs": mean_lost_jobs['mean']},
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative",
                            projectname=projectname, view=view)
    plotter.plot({"costs per ts": mean_costs['mean'],
                  "processed jobs per ts": mean_processed_jobs['mean'],
                  "lost jobs per ts": mean_lost_jobs['mean']},
                 xlabel="timeslot", title=f"[{projectname}] Mean per Timeslot",
                 projectname=projectname, view=view)

    plotter.bar({"job wait time": mean_wait_time['mean']},
                xlabel="timeslot", ylabel="percentage of wait time", title=f"[{projectname}] Mean Job Wait Time",
                projectname=projectname, view=view)
