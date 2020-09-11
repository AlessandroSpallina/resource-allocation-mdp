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


def get_mean_component_costs(raw_stats):
    cps = np.array([d['component_costs_per_timeslot'] for d in raw_stats])
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


def get_mean_wait_time_in_the_queue(raw_stats):
    raw_wt = [d['wait_time_in_the_queue_per_job'] for d in raw_stats]

    max_len = max(len(x) for x in raw_wt)
    wt = []
    for simulation in raw_wt:
        wt.append(simulation + [0] * (max_len - len(simulation)))
    wt = np.array(wt)
    return {
        "mean": wt.mean(axis=0),
        "var": wt.var(axis=0)
    }


def get_mean_wait_time_in_the_system(raw_stats):
    raw_wt = [d['wait_time_in_the_system_per_job'] for d in raw_stats]

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


# # 3 azioni (0,1,2) e 6 stati
# def generate_random_policy(states_num, action_num):
#     rpolicy = []
#     for s in range(states_num):
#         rpolicy.append(random.randint(0, action_num - 1))
#     return tuple(rpolicy)


def generate_conservative_policy(states_num):
    return tuple([1] * states_num)


def generate_smart_conservative_policy(states):
    smpolicy = []
    # if the queue len is bigger than the number of allocated servers -> allocate
    # if the queue len is lower than the number of allocated servers -> deallocate
    # otherwise do nothing
    for state in states:
        if state.k > state.n:
            smpolicy.append(1)
        elif state.k < state.n:
            smpolicy.append(2)
        else:
            smpolicy.append(0)
    return tuple(smpolicy)


# old average, this hides some information
def average_plot_points(data, average_window):
    chunks = np.split(data, average_window)
    averaged = [chunk.mean() for chunk in chunks]

    return np.arange(len(data), step=len(data)/average_window), averaged


def moving_average(data, average_window):
    averaged = np.convolve(data, np.ones((average_window,))/average_window, mode='valid')
    return np.arange(len(data), step=len(data)/averaged.size), averaged


def easy_plot(projectname, stats, max_points_in_plot, view=False):

    cost_per_ts = moving_average(stats['costs_per_timeslot'], max_points_in_plot)
    processed_per_ts = moving_average(stats['processed_jobs_per_timeslot'], max_points_in_plot)
    lost_per_ts = moving_average(stats['lost_jobs_per_timeslot'], max_points_in_plot)
    jobs_in_queue_per_ts = moving_average(stats['jobs_in_queue_per_timeslot'], max_points_in_plot)
    active_server_per_ts = moving_average(stats['active_servers_per_timeslot'], max_points_in_plot)

    job_component_costs = moving_average(stats['component_costs_per_timeslot'][:, 0, 1], max_points_in_plot)
    server_component_costs = moving_average(stats['component_costs_per_timeslot'][:, 1, 1], max_points_in_plot)
    lost_component_costs = moving_average(stats['component_costs_per_timeslot'][:, 2, 1], max_points_in_plot)

    alpha_job_component_costs = moving_average([np.prod(i) for i in stats['component_costs_per_timeslot'][:, 0]], max_points_in_plot)
    beta_server_component_costs = moving_average([np.prod(i) for i in stats['component_costs_per_timeslot'][:, 1]], max_points_in_plot)
    gamma_lost_component_costs = moving_average([np.prod(i) for i in stats['component_costs_per_timeslot'][:, 2]], max_points_in_plot)


    plotter.table([f'{i} jobs' for i in range(len(stats['policy']))],
                  [f'{i} servers' for i in range(len(stats['policy'][0]))],
                  stats['policy'], title=f"{projectname}", projectname=projectname, view=view)

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

    plotter.plot_cumulative(ydata={"jobs in the queue": job_component_costs[1],
                                   "servers": server_component_costs[1],
                                   "lost jobs": lost_component_costs[1]}, xdata=job_component_costs[0],
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative Cost Components",
                            projectname=projectname, view=view)

    plotter.plot_cumulative(ydata={"jobs in the queue": alpha_job_component_costs[1],
                                   "servers": beta_server_component_costs[1],
                                   "lost jobs": gamma_lost_component_costs[1]},
                            xdata=alpha_job_component_costs[0],
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative Cost Components with alpha,beta,gamma",
                            projectname=projectname, view=view)

    # total time is wait time in the system
    plotter.bar(ydata={"job wait time": stats['wait_time_in_the_system_per_job']},
                xlabel="timeslot", ylabel="% of jobs",
                title=f"[{projectname}] Mean Job Wait Time in the System (Total Time)",
                projectname=projectname, view=view)

    plotter.bar(ydata={"job wait time": stats['wait_time_in_the_queue_per_job']},
                xlabel="timeslot", ylabel="% of jobs",
                title=f"[{projectname}] Mean Job Wait Time in the Queue",
                projectname=projectname, view=view)

    plotter.plot_two_scales(jobs_in_queue_per_ts[1], active_server_per_ts[1], xdata=jobs_in_queue_per_ts[0],
                            ylabel1="jobs in queue", ylabel2="active servers", xlabel="timeslot",
                            title=f"[{projectname}] Mean Queue and Servers", projectname=projectname, view=view)


def comparison_plot(projectname, comparison_stats, max_points_in_plot, view=False):

    mdp_cost_per_ts = moving_average(comparison_stats['mdp']['costs_per_timeslot'], max_points_in_plot)
    mdp_processed_per_ts = moving_average(comparison_stats['mdp']['processed_jobs_per_timeslot'], max_points_in_plot)
    mdp_lost_per_ts = moving_average(comparison_stats['mdp']['lost_jobs_per_timeslot'], max_points_in_plot)
    mdp_jobs_in_queue_per_ts = moving_average(comparison_stats['mdp']['jobs_in_queue_per_timeslot'], max_points_in_plot)
    mdp_active_server_per_ts = moving_average(comparison_stats['mdp']['active_servers_per_timeslot'], max_points_in_plot)

    conservative_cost_per_ts = moving_average(comparison_stats['conservative']['costs_per_timeslot'], max_points_in_plot)
    conservative_processed_per_ts = moving_average(comparison_stats['conservative']['processed_jobs_per_timeslot'], max_points_in_plot)
    conservative_lost_per_ts = moving_average(comparison_stats['conservative']['lost_jobs_per_timeslot'], max_points_in_plot)
    conservative_jobs_in_queue_per_ts = moving_average(comparison_stats['conservative']['jobs_in_queue_per_timeslot'], max_points_in_plot)
    conservative_active_server_per_ts = moving_average(comparison_stats['conservative']['active_servers_per_timeslot'], max_points_in_plot)

    smart_conservative_cost_per_ts = moving_average(comparison_stats['smart conservative']['costs_per_timeslot'], max_points_in_plot)
    smart_conservative_processed_per_ts = moving_average(comparison_stats['smart conservative']['processed_jobs_per_timeslot'], max_points_in_plot)
    smart_conservative_lost_per_ts = moving_average(comparison_stats['smart conservative']['lost_jobs_per_timeslot'], max_points_in_plot)
    smart_conservative_jobs_in_queue_per_ts = moving_average(comparison_stats['smart conservative']['jobs_in_queue_per_timeslot'], max_points_in_plot)
    smart_conservative_active_server_per_ts = moving_average(comparison_stats['smart conservative']['active_servers_per_timeslot'], max_points_in_plot)

    plotter.plot_cumulative(ydata={"mdp": mdp_cost_per_ts[1],
                                   "conservative": conservative_cost_per_ts[1],
                                   "smart conservative": smart_conservative_cost_per_ts[1]
                                   }, xdata=mdp_cost_per_ts[0],
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative Costs",
                            projectname=projectname, view=view)

    plotter.plot_cumulative({"mdp": mdp_processed_per_ts[1],
                             "conservative": conservative_processed_per_ts[1],
                             "smart conservative": smart_conservative_processed_per_ts[1]},
                            ylabel="job", xdata=mdp_processed_per_ts[0],
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative Processed Jobs",
                            projectname=projectname, view=view)

    plotter.plot_cumulative({"mdp": mdp_lost_per_ts[1],
                             "conservative": conservative_lost_per_ts[1],
                             "smart conservative": smart_conservative_lost_per_ts[1]},
                            ylabel="job", xdata=mdp_lost_per_ts[0],
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative Lost Jobs",
                            projectname=projectname, view=view)

    plotter.plot(ydata={"mdp": comparison_stats['mdp']['wait_time_in_the_queue_per_job'],
                        "conservative": comparison_stats['conservative']['wait_time_in_the_queue_per_job'],
                        "smart conservative": comparison_stats['smart conservative']['wait_time_in_the_queue_per_job']},
                 xlabel="timeslot",
                 ylabel="% of jobs", title=f"[{projectname}] Mean Job Wait Time in the Queue",
                 projectname=projectname, view=view)

    plotter.plot(ydata={"mdp": comparison_stats['mdp']['wait_time_in_the_system_per_job'],
                        "conservative": comparison_stats['conservative']['wait_time_in_the_system_per_job'],
                        "smart conservative": comparison_stats['smart conservative']['wait_time_in_the_system_per_job']},
                 xlabel="timeslot",
                 ylabel="% of jobs", title=f"[{projectname}] Mean Job Wait Time in the System (Total Time)",
                 projectname=projectname, view=view)

    plotter.plot(ydata={"mdp": mdp_jobs_in_queue_per_ts[1],
                        "conservative": conservative_jobs_in_queue_per_ts[1],
                        "smart conservative": smart_conservative_jobs_in_queue_per_ts[1]},
                 xdata=mdp_jobs_in_queue_per_ts[0], xlabel="timeslot",
                 title=f"[{projectname}] Jobs in queue per Timeslot", projectname=projectname, view=view)

    plotter.plot(ydata={"mdp": mdp_active_server_per_ts[1],
                        "conservative": conservative_active_server_per_ts[1],
                        "smart conservative": smart_conservative_active_server_per_ts[1]},
                 xdata=mdp_active_server_per_ts[0], xlabel="timeslot",
                 title=f"[{projectname}] Active server per Timeslot", projectname=projectname, view=view)
