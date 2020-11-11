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


def plot_cumulative_cost_processed_job(processed_per_ts, lost_per_ts,
                                       title="", save_path="", average_window=10):

    # cost_per_ts = utils.moving_average(cost_per_ts, average_window)
    processed_per_ts = utils.moving_average(processed_per_ts, average_window)
    lost_per_ts = utils.moving_average(lost_per_ts, average_window)

    plot.plot_cumulative(ydata={# "costs": cost_per_ts[1],
                                "processed jobs": processed_per_ts[1],
                                "lost jobs": lost_per_ts[1]},
                         xdata=processed_per_ts[0], xlabel="timeslot",
                         title=f"[{title}] Mean Cumulative", save_path=save_path)


def main(argv):
    DATA_PATH = cli_handler(argv)['data']
    EXPORTED_FILES_PATH = f"../../res/plots/{DATA_PATH.split('/')[-2]}/"
    AVERAGE_WINDOW = 10

    os.makedirs(EXPORTED_FILES_PATH)
    imported_data = utils.import_data(DATA_PATH)

    raw_system_processed_per_ts = [sim_ts['processed_jobs'] for sim_ts in imported_data['simulation_data']]
    system_processed_per_ts = np.sum(raw_system_processed_per_ts, axis=1)
    slices_processed_per_ts = np.hsplit(np.array(raw_system_processed_per_ts), len(raw_system_processed_per_ts[0]))
    for i in range(len(slices_processed_per_ts)):
        slices_processed_per_ts[i] = slices_processed_per_ts[i].reshape(len(slices_processed_per_ts[i]))

    raw_system_lost_per_ts = [sim_ts['lost_jobs'] for sim_ts in imported_data['simulation_data']]
    system_lost_per_ts = np.sum(raw_system_lost_per_ts, axis=1)
    slices_lost_per_ts = np.hsplit(np.array(raw_system_lost_per_ts), len(raw_system_lost_per_ts[0]))
    for i in range(len(slices_lost_per_ts)):
        slices_lost_per_ts[i] = slices_lost_per_ts[i].reshape(len(slices_lost_per_ts[i]))

    plot_cumulative_cost_processed_job(system_processed_per_ts, system_lost_per_ts,
                                       "System MDP", f"{EXPORTED_FILES_PATH}system_cumulative")

    for i in range(len(raw_system_lost_per_ts[0])):
        plot_cumulative_cost_processed_job(slices_processed_per_ts[i], slices_lost_per_ts[i],
                                           f"Slice {i}", f"{EXPORTED_FILES_PATH}slice_{i}_cumulative")


    # print(imported_data['policy'])


if __name__ == '__main__':
    main(sys.argv[1:])
