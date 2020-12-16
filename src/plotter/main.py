# PLOTTER MAIN
# this software il executed by slicing_core passing via argument the path of a result.data

import getopt
import sys
import os
import numpy as np
import copy

import src.plotter.utils as utils
import src.plotter.plot as plotter


def cli_handler(argv):
    USAGE = "main.py -d <dataPath> -w <workingDirectory>"
    to_return = {}
    try:
        opts, args = getopt.getopt(argv, "hd:w:", ["data=", "wdir="])
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
        elif opt in ('-d', '--data'):
            to_return['data'] = arg
        elif opt in ('-w', '--wdir'):
            to_return['wdir'] = arg
    return to_return


def moving_average(data, average_window):
    averaged = np.convolve(data, np.ones((average_window,)) / average_window, mode='valid')
    return np.arange(len(data), step=len(data) / averaged.size), averaged


# TODO: needs to debug with fh
def plot_policy(base_save_path, policy, states):
    if policy[0][0] > 0:  # if here we have vi (mdp) or other policy not based on the timeslot
        with open(f"{base_save_path}policy.csv", "w") as f:
            f.write("state;action\n")
            for policy_i in range(len(policy)):
                line = str([(sub['k'], sub['n']) for sub in states[policy_i]])
                line += f";{policy[policy_i]}"
                f.write(f"{line}\n")
    else:  # fh
        previous_policy = []
        ts_len = len(policy[0])
        for ts in range(ts_len):
            policy_ts = np.array(policy)[:, ts]
            if not np.array_equal(policy_ts, previous_policy):
                # plot policy_ts!
                with open(f"{base_save_path}ts-{ts}.csv", "w") as f:
                    f.write("state;action\n")
                    for s in range(len(policy_ts)):
                        line = str([(sub['k'], sub['n']) for sub in states[s]])
                        line += f";{policy_ts[s]}"
                        f.write(f"{line}\n")
                previous_policy = policy_ts


# Plotting stuff related to simulation results
def plot_results(plot_identifier, base_save_path, stats, window_average):
    active_servers_per_ts = moving_average(stats['active_servers'], window_average)
    jobs_in_queue_per_ts = moving_average(stats['jobs_in_queue'], window_average)
    lost_jobs_per_ts = moving_average(stats['lost_jobs'], window_average)
    processed_jobs_per_ts = moving_average(stats['processed_jobs'], window_average)
    cost_per_ts = moving_average(stats['cost'], window_average)
    wait_time_in_the_queue = stats['wait_time_in_the_queue']
    wait_time_in_the_system = stats['wait_time_in_the_system']

    plotter.plot_cumulative(ydata={"costs": cost_per_ts[1],
                                   "processed jobs": processed_jobs_per_ts[1],
                                   "lost jobs": lost_jobs_per_ts[1]},
                            xdata=cost_per_ts[0], xlabel="timeslot", title=f"[{plot_identifier}] Mean Cumulative",
                            save_path=f"{base_save_path}cumulative")

    plotter.plot(ydata={"costs per ts": cost_per_ts[1],
                        "processed jobs per ts": processed_jobs_per_ts[1],
                        "lost jobs per ts": lost_jobs_per_ts[1]},
                 xdata=cost_per_ts[0], xlabel="timeslot", title=f"[{plot_identifier}] Mean per Timeslot",
                 save_path=f"{base_save_path}per_ts")

    # total time is wait time in the system
    plotter.bar(ydata={"job wait time": wait_time_in_the_system},
                xlabel="timeslot", ylabel="% of jobs",
                title=f"[{plot_identifier}] Mean Job Wait Time in the System (Total Time)",
                save_path=f"{base_save_path}wait_time_in_system")

    plotter.bar(ydata={"job wait time": wait_time_in_the_queue},
                xlabel="timeslot", ylabel="% of jobs",
                title=f"[{plot_identifier}] Mean Job Wait Time in the Queue",
                save_path=f"{base_save_path}wait_time_in_queue")

    plotter.plot_two_scales(jobs_in_queue_per_ts[1], active_servers_per_ts[1], xdata=jobs_in_queue_per_ts[0],
                            ylabel1="jobs in queue", ylabel2="active servers", xlabel="timeslot",
                            title=f"[{plot_identifier}] Mean Queue and Servers",
                            save_path=f"{base_save_path}jobs_in_queue_vs_active_servers")


def plot_slices_configs(plot_identifier, base_save_path, configs):
    plotter.bar(ydata={"arrivals": configs['arrivals_histogram']},
                xlabel="number of jobs", ylabel="% of probability",
                title=f"[{plot_identifier}] Arrivals Histogram",
                save_path=f"{base_save_path}arrivals_histogram")

    plotter.bar(ydata={"job wait time": configs['server_capacity_histogram']},
                xlabel="number of jobs", ylabel="% of probability",
                title=f"[{plot_identifier}] Server Capacity Histogram",
                save_path=f"{base_save_path}server_capacity_histogram")


def split_stats_per_slice(stats):
    stats_per_slice = [{} for _ in range(len(stats['active_servers'][0]))]

    for key in stats:
        for ts in stats[key]:
            for slice_i in range(len(ts)):
                if not key in stats_per_slice[slice_i]:
                    stats_per_slice[slice_i][key] = []
                stats_per_slice[slice_i][key].append(ts[slice_i])

    return stats_per_slice


def merge_stats_for_system_pow(stats):
    merged = copy.deepcopy(stats)

    for key in stats:
        for ts_i in range(len(stats[key])):
            merged[key][ts_i] = sum(stats[key][ts_i])

    return merged


def main(argv):
    # ---- CLI ARGS HANDLING -----------------------
    cli_args = cli_handler(argv)
    if 'wdir' in cli_args:
        os.chdir(cli_args['wdir'])
        print(f"changed working dir to {os.getcwd()}")
    # ---------------------------------------------

    DATA_PATH = cli_args['data']
    EXPORTED_FILES_PATH = f"../../res/plots/{DATA_PATH.split('/')[-2]}/"
    AVERAGE_WINDOW = 200

    print(f"PLOTTER: Current directory {os.getcwd()}")

    # ---- CREATING STUFF ON FILE SYSTEM -------------------
    os.makedirs(EXPORTED_FILES_PATH)
    if os.name == 'nt':  # we are on windows (symbolic link are not working well using native python)
        os.system(f'mklink /D \"{EXPORTED_FILES_PATH}raw_results\" \"{"/".join(DATA_PATH.split("/")[:-1])}\"')
    else:
        os.system(f'ln -s \"{"/".join(DATA_PATH.split("/")[:-1])}\" \"{EXPORTED_FILES_PATH}raw_results\"')
    # ------------------------------------------------------

    imported_data = utils.import_data(DATA_PATH)

    for result in imported_data:
        result_base_path = f"{EXPORTED_FILES_PATH}{result['name']}/"
        os.makedirs(result_base_path)

        stats_per_slice = split_stats_per_slice(result['environment_data'])

        for i in range(len(stats_per_slice)):
            os.makedirs(f"{result_base_path}slice-{i}")
            plot_slices_configs(f"slice-{i}", f"{result_base_path}slice-{i}/", result['slices'][i])
            plot_results(f"slice-{i}", f"{result_base_path}slice-{i}/", stats_per_slice[i], AVERAGE_WINDOW)

        merged_per_system = merge_stats_for_system_pow(result['environment_data'])
        os.makedirs(f"{result_base_path}system_pow")
        plot_results(f"system-pow", f"{result_base_path}system_pow/", merged_per_system, AVERAGE_WINDOW)
        plot_policy(f"{result_base_path}system_pow/", result['policy'], result['states'])


if __name__ == '__main__':
    main(sys.argv[1:])
