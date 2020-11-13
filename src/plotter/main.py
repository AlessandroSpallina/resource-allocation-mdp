# PLOTTER MAIN
# this software il executed by slicing_core passing via argument the path of a result.data

import getopt
import sys
import os
import numpy as np

import src.plotter.utils as utils
import src.plotter.plot as plot


def cli_handler(argv):
    USAGE = "main.py -d <dataPath>"
    to_return = {}
    try:
        opts, args = getopt.getopt(argv, "hd:", ["data="])
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
        elif opt in ('-d', '--data'):
            to_return['data'] = arg
    return to_return

# TODO: da fixare
# def plot_policy(policy, title="", save_path=""):
#     try:
#         if len(policy[0]) > 0:  # if we are here the policy is a matrix (fh)
#             previous_policy = []
#             for i in range(len(policy)):
#                 if policy[i] != previous_policy:
#                     previous_policy = policy[i]
#                     plot.table([f'{i} jobs' for i in range(len(previous_policy))],
#                                [f'{i} servers' for i in range(len(previous_policy[0]))],
#                                previous_policy, title=f"{title} (ts {i})", save_path=save_path)
#     except TypeError:
#         plot.table([f'{i} jobs' for i in range(len(policy))],
#                    [f'{i} servers' for i in range(len(policy[0]))],
#                    policy, title=f"{title}", save_path=save_path)


def plot_cumulative_cost_processed_job(cost_per_ts, processed_per_ts, lost_per_ts,
                                       title="", save_path="", average_window=10):

    cost_per_ts = utils.moving_average(cost_per_ts, average_window)
    processed_per_ts = utils.moving_average(processed_per_ts, average_window)
    lost_per_ts = utils.moving_average(lost_per_ts, average_window)

    plot.plot_cumulative(ydata={"costs": cost_per_ts[1],
                                "processed jobs": processed_per_ts[1],
                                "lost jobs": lost_per_ts[1]},
                         xdata=cost_per_ts[0], xlabel="timeslot",
                         title=f"[{title}] Mean Cumulative", save_path=save_path)


def plot_per_ts_cost_processed_job(cost_per_ts, processed_per_ts, lost_per_ts,
                                   title="", save_path="", average_window=10):

    cost_per_ts = utils.moving_average(cost_per_ts, average_window)
    processed_per_ts = utils.moving_average(processed_per_ts, average_window)
    lost_per_ts = utils.moving_average(lost_per_ts, average_window)

    plot.plot(ydata={"costs": cost_per_ts[1],
                     "processed jobs": processed_per_ts[1],
                     "lost jobs": lost_per_ts[1]},
              xdata=cost_per_ts[0], xlabel="timeslot",
              title=f"[{title}] Mean", save_path=save_path)


def plot_per_ts_job_in_queue_vs_active_server(state, title="", save_path="", average_window=10):
    k = []
    n = []
    if type(state) == list:  # we are plotting a system pow
        for ts in state:
            k.append(sum([s['k'] for s in ts]))
            n.append(sum([s['n'] for s in ts]))
    else:
        for ts in state:
            k.append(ts['k'])
            n.append(ts['n'])

    # k = utils.moving_average(k, average_window)
    # n = utils.moving_average(n, average_window)

    # plot.plot_two_scales(data1=k[1], ylabel1="jobs in the queue",
    #                      data2=n[1], ylabel2="active servers",
    #                      xdata=k[0], xlabel="timeslot", title=title, save_path=save_path)
    plot.plot_two_scales(data1=k, ylabel1="jobs in the queue",
                         data2=n, ylabel2="active servers",
                         xlabel="timeslot", title=title, save_path=save_path)



def get_stat_system_slice_pow(stat, keyword):
    raw_system_stat_per_ts = [sim_ts[keyword] for sim_ts in stat]
    system_stat_per_ts = np.sum(raw_system_stat_per_ts, axis=1)
    slices_stat_per_ts = np.hsplit(np.array(raw_system_stat_per_ts), len(raw_system_stat_per_ts[0]))
    for i in range(len(slices_stat_per_ts)):
        slices_stat_per_ts[i] = slices_stat_per_ts[i].reshape(len(slices_stat_per_ts[i]))
    return {'system_pow': system_stat_per_ts, 'slices_pow': slices_stat_per_ts}


def get_state_system_slice_pow(stat):
    system_stat_per_ts = [sim_ts['state'] for sim_ts in stat]
    slices_stat_per_ts = np.hsplit(np.array(system_stat_per_ts), len(system_stat_per_ts[0]))
    for i in range(len(slices_stat_per_ts)):
        slices_stat_per_ts[i] = slices_stat_per_ts[i].reshape(len(slices_stat_per_ts[i]))
    return {'system_pow': system_stat_per_ts, 'slices_pow': slices_stat_per_ts}


def main(argv):
    DATA_PATH = cli_handler(argv)['data']
    EXPORTED_FILES_PATH = f"../../res/plots/{DATA_PATH.split('/')[-2]}/"
    AVERAGE_WINDOW = 200

    # ---- CREATING STUFF ON FILE SYSTEM -------------------
    os.makedirs(EXPORTED_FILES_PATH)
    if os.name == 'nt':  # we are on windows (symbolic link are not working well using native python)
        os.system(f'mklink /D \"{EXPORTED_FILES_PATH}raw_results\" \"{"/".join(DATA_PATH.split("/")[:-1])}\"')
    else:
        os.symlink(f"{EXPORTED_FILES_PATH}raw_results", "/".join(DATA_PATH.split("/")[:-1]), True)
    # ------------------------------------------------------

    imported_data = utils.import_data(DATA_PATH)

    # ---- PROCESSING --------------------------------------
    cost = get_stat_system_slice_pow(imported_data['environment_data'], 'cost')
    processed = get_stat_system_slice_pow(imported_data['environment_data'], 'processed_jobs')
    lost = get_stat_system_slice_pow(imported_data['environment_data'], 'lost_jobs')
    state = get_state_system_slice_pow(imported_data['environment_data'])
    # ------------------------------------------------------

    # ---- PLOTTING ----------------------------------------
    plot_cumulative_cost_processed_job(cost['system_pow'], processed['system_pow'], lost['system_pow'],
                                       "System MDP", f"{EXPORTED_FILES_PATH}system_cumulative", AVERAGE_WINDOW)
    plot_per_ts_cost_processed_job(cost['system_pow'], processed['system_pow'], lost['system_pow'],
                                   "System MDP", f"{EXPORTED_FILES_PATH}system_per_ts", AVERAGE_WINDOW)
    plot_per_ts_job_in_queue_vs_active_server(state['system_pow'], "System MDP",
                                              f"{EXPORTED_FILES_PATH}jobs_vs_servers_per_ts", AVERAGE_WINDOW)

    for i in range(len(cost['slices_pow'])):  # for each slice
        plot_cumulative_cost_processed_job(cost['slices_pow'][i], processed['slices_pow'][i], lost['slices_pow'][i],
                                           f"Slice {i}", f"{EXPORTED_FILES_PATH}slice_{i}_cumulative", AVERAGE_WINDOW)
        plot_per_ts_cost_processed_job(cost['slices_pow'][i], processed['slices_pow'][i], lost['slices_pow'][i],
                                       f"Slice {i}", f"{EXPORTED_FILES_PATH}slice_{i}_per_ts", AVERAGE_WINDOW)
        plot_per_ts_job_in_queue_vs_active_server(state['slices_pow'][i], f"Slice {i}",
                                                  f"{EXPORTED_FILES_PATH}jobs_vs_servers_{i}_per_ts", AVERAGE_WINDOW)

    # -----------------------------------------------------


if __name__ == '__main__':
    main(sys.argv[1:])
