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
def plot_slice_results(plot_identifier, base_save_path, stats, window_average, is_system_pow=False):
    active_servers_per_ts = moving_average(stats['active_servers'], window_average)
    jobs_in_queue_per_ts = moving_average(stats['jobs_in_queue'], window_average)
    lost_jobs_per_ts = moving_average(stats['lost_jobs'], window_average)
    processed_jobs_per_ts = moving_average(stats['processed_jobs'], window_average)
    cost_per_ts = moving_average(stats['cost'], window_average)

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

    plotter.plot_two_scales(jobs_in_queue_per_ts[1], active_servers_per_ts[1], xdata=jobs_in_queue_per_ts[0],
                            ylabel1="jobs in queue", ylabel2="active servers", xlabel="timeslot",
                            title=f"[{plot_identifier}] Mean Queue and Servers",
                            save_path=f"{base_save_path}jobs_in_queue_vs_active_servers")

    if not is_system_pow:
        wait_time_in_the_queue = stats['wait_time_in_the_queue']
        wait_time_in_the_system = stats['wait_time_in_the_system']

        # total time is wait time in the system
        plotter.bar(ydata={"job wait time": wait_time_in_the_system},
                    xlabel="timeslot", ylabel="% of jobs",
                    title=f"[{plot_identifier}] Mean Job Wait Time in the System (Total Time)",
                    save_path=f"{base_save_path}wait_time_in_system")

        plotter.bar(ydata={"job wait time": wait_time_in_the_queue},
                    xlabel="timeslot", ylabel="% of jobs",
                    title=f"[{plot_identifier}] Mean Job Wait Time in the Queue",
                    save_path=f"{base_save_path}wait_time_in_queue")


def plot_slice_comparison(plot_identifier, base_save_path, stats, window_average):
    cost_per_ts = {}
    processed_per_ts = {}
    lost_per_ts = {}
    jobs_in_queue_per_ts = {}
    active_server_per_ts = {}

    wait_time_in_the_queue = {}
    wait_time_in_the_system = {}

    xdata = []

    for slice_i in stats:
        xdata = moving_average(slice_i['cost'], window_average)[0]

        cost_per_ts[slice_i['policy_name']] = moving_average(slice_i['cost'], window_average)[1]
        processed_per_ts[slice_i['policy_name']] = moving_average(slice_i['processed_jobs'], window_average)[1]
        lost_per_ts[f"{slice_i['policy_name']} ({sum(slice_i['lost_jobs']) / sum(slice_i['incoming_jobs'])}%)"] = \
            moving_average(slice_i['lost_jobs'], window_average)[1]
        jobs_in_queue_per_ts[slice_i['policy_name']] = moving_average(slice_i['jobs_in_queue'], window_average)[1]
        active_server_per_ts[slice_i['policy_name']] = moving_average(slice_i['active_servers'], window_average)[1]

        wait_time_in_the_queue[slice_i['policy_name']] = slice_i['wait_time_in_the_queue']
        wait_time_in_the_system[slice_i['policy_name']] = slice_i['wait_time_in_the_system']

    plotter.plot_cumulative(ydata=cost_per_ts, xdata=xdata,
                            xlabel="timeslot", title=f"[{plot_identifier}] Mean Cumulative Costs",
                            save_path=f"{base_save_path}cumulative_costs")

    plotter.plot_cumulative(processed_per_ts,
                            ylabel="job", xdata=xdata,
                            xlabel="timeslot", title=f"[{plot_identifier}] Mean Cumulative Processed Jobs",
                            save_path=f"{base_save_path}cumulative_processed_jobs")

    plotter.plot_cumulative(lost_per_ts,
                            ylabel="job", xdata=xdata,
                            xlabel="timeslot", title=f"[{plot_identifier}] Mean Cumulative Lost Jobs",
                            save_path=f"{base_save_path}cumulative_lost_jobs")

    plotter.bar(ydata=wait_time_in_the_queue,
                xlabel="timeslot",
                ylabel="% of jobs", title=f"[{plot_identifier}] Mean Job Wait Time in the Queue",
                save_path=f"{base_save_path}wait_time_in_queue")

    plotter.bar(ydata=wait_time_in_the_system,
                xlabel="timeslot",
                ylabel="% of jobs", title=f"[{plot_identifier}] Mean Job Wait Time in the System (Total Time)",
                save_path=f"{base_save_path}wait_time_in_system")

    plotter.plot(ydata=jobs_in_queue_per_ts,
                 xdata=xdata, xlabel="timeslot",
                 title=f"[{plot_identifier}] Jobs in queue per Timeslot", save_path=f"{base_save_path}jobs_in_queue")

    plotter.plot(ydata=active_server_per_ts,
                 xdata=xdata, xlabel="timeslot",
                 title=f"[{plot_identifier}] Active server per Timeslot", save_path=f"{base_save_path}active_servers")


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


# deletes 0s from wait_time in the queue and in the system, these 0s are without meaning here (not true in other stats!)
def filter_stats_per_slice(stats_per_slice):
    # wait_time_in_the_queue_per_slice = stats_per_slice['wait_time_in_the_queue']
    # wait_time_in_the_system_per_slice = stats_per_slice['wait_time_in_the_queue']
    for s_index in range(len(stats_per_slice)):
        to_filter_queue = stats_per_slice[s_index]['wait_time_in_the_queue']
        for element in reversed(to_filter_queue):
            if element == 0:
                to_filter_queue.pop()
            else:
                break

        to_filter_system = stats_per_slice[s_index]['wait_time_in_the_system']
        for element in reversed(to_filter_system):
            if element == 0:
                to_filter_system.pop()
            else:
                break

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

    # the number of slices is the same for each policy, because the config is the same
    slice_comparison_stats = {i: [] for i in range(len(imported_data[0]['slices']))}

    for result in imported_data:
        result_base_path = f"{EXPORTED_FILES_PATH}{result['name']}/"
        os.makedirs(result_base_path)

        stats_per_slice = filter_stats_per_slice(split_stats_per_slice(result['environment_data']))

        for i in range(len(stats_per_slice)):
            os.makedirs(f"{result_base_path}slice-{i}")
            plot_slices_configs(f"slice-{i}", f"{result_base_path}slice-{i}/", result['slices'][i])
            plot_slice_results(f"slice-{i}", f"{result_base_path}slice-{i}/", stats_per_slice[i], AVERAGE_WINDOW)
            tmp = stats_per_slice[i]
            tmp['policy_name'] = result['name']
            slice_comparison_stats[i].append(stats_per_slice[i])

        merged_per_system = merge_stats_for_system_pow(result['environment_data'])
        os.makedirs(f"{result_base_path}system_pow")
        plot_slice_results(f"system-pow",
                           f"{result_base_path}system_pow/", merged_per_system, AVERAGE_WINDOW, is_system_pow=True)
        plot_policy(f"{result_base_path}system_pow/", result['policy'], result['states'])

    os.makedirs(f"{EXPORTED_FILES_PATH}comparison/")
    for slice_i in range(len(slice_comparison_stats)):
        os.makedirs(f"{EXPORTED_FILES_PATH}/comparison/slice-{slice_i}")
        plot_slice_comparison(f"comparison slice-{slice_i}",
                              f"{EXPORTED_FILES_PATH}/comparison/slice-{slice_i}/",
                              slice_comparison_stats[slice_i], AVERAGE_WINDOW)


if __name__ == '__main__':
    main(sys.argv[1:])
