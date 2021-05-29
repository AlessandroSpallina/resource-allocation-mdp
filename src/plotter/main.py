# PLOTTER MAIN
# this software is executed by slicing_core passing via argument the path of a result.data

import getopt
import sys
import os
import numpy as np
import copy
import pandas as pd
import functools
import operator
import itertools

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
    if type(data[0]) == list:
        return_values_timeline = None
        return_values = []
        to_average = np.hsplit(np.array(data), len(data[0]))
        for subdata in to_average:
            subdata = list(itertools.chain(*subdata))
            tmp_timeline, tmp_values = moving_average(subdata, average_window)
            return_values_timeline = tmp_timeline
            return_values.append(tmp_values)
        return return_values_timeline, return_values

    averaged = np.convolve(data, np.ones((average_window,)) / average_window, mode='valid')
    timeline = np.arange(len(data), step=len(data) / averaged.size)
    if timeline.size > averaged.size:
        timeline = np.resize(timeline, -(timeline.size - averaged.size))
    return timeline, averaged


# TODO: needs to debug with fh
def plot_policy(base_save_path, policy, states):
    if policy[0][0] > 0:  # if here we have vi (mdp) or other policy not based on the timeslot
        with open(f"{base_save_path}policy.csv", "w") as f:
            f.write("state;action\n")
            for policy_i in range(len(policy)):
                line = str([(sub['k'], sub['n']) for sub in states[policy_i]])
                line += f";{policy[policy_i]}"
                f.write(f"{line}\n")

        single_slice_states = [[] for _ in range(len(states[0]))]
        single_slice_actions = [[] for _ in range(len(states[0]))]
        for multi_state_index in range(len(states)):
            multi_state = states[multi_state_index]
            multi_action = policy[multi_state_index]

            for slice_index in range(len(multi_state)):
                state_tmp = multi_state[slice_index]
                action_tmp = multi_action[slice_index]

                if state_tmp not in single_slice_states[slice_index]:
                    single_slice_states[slice_index].append(state_tmp)
                    single_slice_actions[slice_index].append([action_tmp])
                else:
                    # cerca l'index dello stato piccolo e verifica se c'è già action in policy
                    state_i = single_slice_states[slice_index].index(state_tmp)
                    if action_tmp not in single_slice_actions[slice_index][state_i]:
                        single_slice_actions[slice_index][state_i].append(action_tmp)
                        single_slice_actions[slice_index][state_i].sort()

        dataframes_dictionaries = \
            [{'job': [], 'server': [], 'action': []} if _ == 0 else
             [{'job': [], 'server': [], 'action': []}, {'job': [], 'server': [], 'action': []}]
             for _ in range(len(states[0]))]

        # calculating dataframes
        for slice_index in range(len(single_slice_states)):
            for state_index in range(len(single_slice_states[slice_index])):
                state_tmp = single_slice_states[slice_index][state_index]
                actions_tmp = single_slice_actions[slice_index][state_index]

                if slice_index == 0:  # if == 0 there is no min and max, but only one policy
                    dataframes_dictionaries[slice_index]['job'].append(state_tmp['k'])
                    dataframes_dictionaries[slice_index]['server'].append(state_tmp['n'])
                    dataframes_dictionaries[slice_index]['action'].append(actions_tmp[0])
                else:
                    act_min = actions_tmp[0]
                    act_max = actions_tmp[-1]

                    dataframes_dictionaries[slice_index][0]['job'].append(state_tmp['k'])
                    dataframes_dictionaries[slice_index][0]['server'].append(state_tmp['n'])
                    dataframes_dictionaries[slice_index][0]['action'].append(act_min)

                    dataframes_dictionaries[slice_index][1]['job'].append(state_tmp['k'])
                    dataframes_dictionaries[slice_index][1]['server'].append(state_tmp['n'])
                    dataframes_dictionaries[slice_index][1]['action'].append(act_max)

        # doing heatmap
        for i in range(len(dataframes_dictionaries)):
            if i == 0:  # slice with index 0 need only 1 heatmap
                df = pd.DataFrame(dataframes_dictionaries[i])
                plotter.plot_heatmap(df, title=f"Slice-{i} Policy", save_path=f"{base_save_path}slice-{i}_policy.png")
            else:
                for d in range(len(dataframes_dictionaries[i])):
                    df = pd.DataFrame(dataframes_dictionaries[i][d])
                    plotter.plot_heatmap(df, title=f"Slice-{i} Policy {'min' if d==0 else 'max'}",
                                         save_path=f"{base_save_path}slice-{i}_{'min' if d==0 else 'max'}_policy.png")

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


def plot_slice_results_confidence(plot_identifier, base_save_path, stats, slice_index, window_average, is_system_pow=False):
    slice_stats = []

    for sim in stats:
        slice_stats.append(sim[slice_index])

    ts_in_simulation = moving_average(slice_stats[0]['active_servers'], window_average)[0].tolist()
    #ts_in_simulation = list(range(len(slice_stats[0]['active_servers'])))
    timeslot_tmp = ts_in_simulation * len(slice_stats)
    simulation_tmp = functools.reduce(operator.iconcat, [[sim_id]*len(ts_in_simulation) for sim_id in range(len(slice_stats))], [])

    active_servers_tmp = [moving_average(sim['active_servers'], window_average)[1] for sim in slice_stats]
    jobs_in_queue_tmp = [moving_average(sim['jobs_in_queue'], window_average)[1] for sim in slice_stats]
    lost_jobs_tmp = [moving_average(sim['lost_jobs'], window_average)[1] for sim in slice_stats]
    processed_jobs_tmp = [moving_average(sim['processed_jobs'], window_average)[1] for sim in slice_stats]
    cost_tmp = [moving_average(sim['cost'], window_average)[1] for sim in slice_stats]

    active_servers_tmp = functools.reduce(operator.iconcat, active_servers_tmp, [])
    jobs_in_queue_tmp = functools.reduce(operator.iconcat, jobs_in_queue_tmp, [])
    lost_jobs_tmp = functools.reduce(operator.iconcat, lost_jobs_tmp, [])
    processed_jobs_tmp = functools.reduce(operator.iconcat, processed_jobs_tmp, [])
    cost_tmp = functools.reduce(operator.iconcat, cost_tmp, [])

    # ----
    active_servers_df = pd.DataFrame({'active_servers': active_servers_tmp,
                                      'timeslot': timeslot_tmp,
                                      'simulation': simulation_tmp})

    plotter.plot_confidence(active_servers_df,
                            title=f"[{plot_identifier}] Active servers",
                            xlabel="timeslot", ylabel="active_servers",
                            save_path=f"{base_save_path}active_servers_confidence-wa{window_average}")
    # ----
    jobs_in_queue_tmp = pd.DataFrame({'jobs_in_queue': jobs_in_queue_tmp,
                                      'timeslot': timeslot_tmp,
                                      'simulation': simulation_tmp})

    plotter.plot_confidence(jobs_in_queue_tmp,
                            title=f"[{plot_identifier}] Jobs in queue",
                            xlabel="timeslot", ylabel="jobs_in_queue",
                            save_path=f"{base_save_path}jobs_in_queue_confidence-wa{window_average}")
    # ----
    lost_jobs_tmp = pd.DataFrame({'lost_jobs': lost_jobs_tmp,
                                  'timeslot': timeslot_tmp,
                                  'simulation': simulation_tmp})

    plotter.plot_confidence(lost_jobs_tmp,
                            title=f"[{plot_identifier}] Lost jobs",
                            xlabel="timeslot", ylabel="lost_jobs",
                            save_path=f"{base_save_path}lost_jobs_confidence-wa{window_average}")
    # ----
    processed_jobs_tmp = pd.DataFrame({'processed_jobs': processed_jobs_tmp,
                                       'timeslot': timeslot_tmp,
                                       'simulation': simulation_tmp})

    plotter.plot_confidence(processed_jobs_tmp,
                            title=f"[{plot_identifier}] Processed jobs",
                            xlabel="timeslot", ylabel="processed_jobs",
                            save_path=f"{base_save_path}processed_jobs_confidence-wa{window_average}")
    # ----
    cost_tmp = pd.DataFrame({'cost': cost_tmp,
                             'timeslot': timeslot_tmp,
                             'simulation': simulation_tmp})

    plotter.plot_confidence(cost_tmp,
                            title=f"[{plot_identifier}] Costs",
                            xlabel="timeslot", ylabel="cost",
                            save_path=f"{base_save_path}cost_confidence-wa{window_average}")
    # ----


# Plotting stuff related to simulation results
def plot_slice_results(plot_identifier, base_save_path, stats, window_average, is_system_pow=False):
    active_servers_per_ts = moving_average(stats['active_servers'], window_average)
    jobs_in_queue_per_ts = moving_average(stats['jobs_in_queue'], window_average)
    lost_jobs_per_ts = moving_average(stats['lost_jobs'], window_average)
    processed_jobs_per_ts = moving_average(stats['processed_jobs'], window_average)
    cost_per_ts = moving_average(stats['cost'], window_average)
    cost_component_per_ts = moving_average(stats['cost_component'], window_average)

    # plotter.plot_cumulative(ydata={"costs": cost_per_ts[1],
    #                                "processed jobs": processed_jobs_per_ts[1],
    #                                "lost jobs": lost_jobs_per_ts[1]},
    #                         xdata=cost_per_ts[0], xlabel="timeslot", title=f"[{plot_identifier}] Mean Cumulative",
    #                         save_path=f"{base_save_path}cumulative-wa{window_average}")

    plotter.plot(ydata={"costs per ts": cost_per_ts[1]},
                 xdata=cost_per_ts[0], xlabel="timeslot", title=f"[{plot_identifier}] Mean costs per Timeslot",
                 save_path=f"{base_save_path}per_ts_costs-wa{window_average}")

    plotter.plot(ydata={
        "job in the queue": cost_component_per_ts[1][0],
        "running server": cost_component_per_ts[1][1],
        "lost jobs": cost_component_per_ts[1][2],
        "allocating server": cost_component_per_ts[1][3],
        "deallocating server": cost_component_per_ts[1][4]
    },
                 xdata=cost_per_ts[0], xlabel="timeslot", ylabel="cost (usd)",
                 title=f"[{plot_identifier}] Mean cost components per Timeslot",
                 save_path=f"{base_save_path}per_ts_cost-components-wa{window_average}")

    plotter.plot_cumulative(ydata={
        "job in the queue": cost_component_per_ts[1][0],
        "running server": cost_component_per_ts[1][1],
        "lost jobs": cost_component_per_ts[1][2],
        "allocating server": cost_component_per_ts[1][3],
        "deallocating server": cost_component_per_ts[1][4]
    },
                            xdata=cost_per_ts[0], xlabel="timeslot", ylabel="cost (usd)",
                            title=f"[{plot_identifier}] Mean cumulative cost components",
                            save_path=f"{base_save_path}cumulative-cost-components-wa{window_average}")

    plotter.plot(ydata={"processed jobs per ts": processed_jobs_per_ts[1]},
                 xdata=cost_per_ts[0], xlabel="timeslot", title=f"[{plot_identifier}] Mean processed per Timeslot",
                 save_path=f"{base_save_path}per_ts_processed-wa{window_average}")

    plotter.plot(ydata={"lost jobs per ts": lost_jobs_per_ts[1]},
                 xdata=cost_per_ts[0], xlabel="timeslot", title=f"[{plot_identifier}] Mean lost per Timeslot",
                 save_path=f"{base_save_path}per_ts_lost-wa{window_average}")

    plotter.plot_two_scales(jobs_in_queue_per_ts[1], active_servers_per_ts[1], xdata=jobs_in_queue_per_ts[0],
                            ylabel1="requests in queue", ylabel2="active servers", xlabel="timeslot",
                            title=f"[{plot_identifier}] Mean Queue and Servers",
                            save_path=f"{base_save_path}jobs_in_queue_vs_active_servers-wa{window_average}")

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
        lost_per_ts[f"{slice_i['policy_name']} ({round((sum(slice_i['lost_jobs']) / sum(slice_i['incoming_jobs'])) * 100, 4)}%)"]\
            = moving_average(slice_i['lost_jobs'], window_average)[1]
        jobs_in_queue_per_ts[slice_i['policy_name']] = moving_average(slice_i['jobs_in_queue'], window_average)[1]
        active_server_per_ts[slice_i['policy_name']] = moving_average(slice_i['active_servers'], window_average)[1]

        wait_time_in_the_queue[f"{slice_i['policy_name']} " \
                               f"(avg. {round(np.average(np.array(slice_i['jobs_in_queue'])) / np.average(np.array(slice_i['incoming_jobs'])), 4)} ts)"] \
            = slice_i['wait_time_in_the_queue']
        wait_time_in_the_system[f"{slice_i['policy_name']} " \
                                f"(avg. {round(np.average(np.array(slice_i['jobs_in_system'])) / np.average(np.array(slice_i['incoming_jobs'])), 4)} ts)"] \
            = slice_i['wait_time_in_the_system']

    plotter.plot_cumulative(ydata=cost_per_ts, xdata=xdata,
                            xlabel="timeslot", title=f"[{plot_identifier}] Mean Cumulative Costs",
                            save_path=f"{base_save_path}cumulative_costs-wa{window_average}")

    plotter.plot_cumulative(processed_per_ts,
                            ylabel="job", xdata=xdata,
                            xlabel="timeslot", title=f"[{plot_identifier}] Mean Cumulative Processed Jobs",
                            save_path=f"{base_save_path}cumulative_processed_jobs-wa{window_average}")

    plotter.plot_cumulative(lost_per_ts,
                            ylabel="job", xdata=xdata,
                            xlabel="timeslot", title=f"[{plot_identifier}] Mean Cumulative Lost Jobs",
                            save_path=f"{base_save_path}cumulative_lost_jobs-wa{window_average}")

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
                 title=f"[{plot_identifier}] Jobs in queue per Timeslot", save_path=f"{base_save_path}jobs_in_queue-wa{window_average}")

    plotter.plot(ydata=active_server_per_ts,
                 xdata=xdata, xlabel="timeslot",
                 title=f"[{plot_identifier}] Active server per Timeslot", save_path=f"{base_save_path}active_servers-wa{window_average}")


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


# returns [sim1, sim2, ..., simN]
# in which simN = [slice0, ..., sliceN]
# in which slice0 = {'active_servers': ..., 'other_stuff': ...}
def split_raw_stats_per_slice(stats):
    raw_stats_per_slice = \
        [[{} for i in range(len(stats['active_servers'][0][0]))] for _ in range(len(stats['active_servers']))]

    for sim in range(len(raw_stats_per_slice)):
        for key in stats:
            for ts in stats[key][sim]:
                for slice_i in range(len(ts)):
                    if not key in raw_stats_per_slice[sim][slice_i]:
                        raw_stats_per_slice[sim][slice_i][key] = []
                    raw_stats_per_slice[sim][slice_i][key].append(ts[slice_i])

    return raw_stats_per_slice


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
        merged[key] = np.sum(np.array(merged[key]), axis=1).tolist()
        # for ts_i in range(len(stats[key])):
        #     merged[key][ts_i] = sum(stats[key][ts_i])

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
    #AVERAGE_WINDOWS = [10, 90, 170, 250]
    AVERAGE_WINDOWS = [250, 300, 350]

    print(f"PLOTTER: Current directory {os.getcwd()}")

    # ---- CREATING STUFF ON FILE SYSTEM -------------------
    os.makedirs(EXPORTED_FILES_PATH)
    if os.name == 'nt':  # we are on windows (symbolic link are not working well using native python)
        os.system(f'mklink /D \"{EXPORTED_FILES_PATH}raw_results\" \"{"/".join(DATA_PATH.split("/")[:-1])}\"')
    else:
        os.system(f'ln -s \"{"/".join(DATA_PATH.split("/")[:-1])}\" \"{EXPORTED_FILES_PATH}raw_results\"')
    # ------------------------------------------------------

    imported_data = utils.import_serialized_data(DATA_PATH)

    # the number of slices is the same for each policy, because the config is the same
    slice_comparison_stats = {i: [] for i in range(len(imported_data[0]['slices']))}

    for result in imported_data:
        result_base_path = f"{EXPORTED_FILES_PATH}{result['name']}/"
        os.makedirs(result_base_path)

        stats_per_slice = filter_stats_per_slice(split_stats_per_slice(result['environment_data']))

        # raw_stats_per_slice = split_raw_stats_per_slice(result['environment_data_raw'])

        for i in range(len(stats_per_slice)):
            slice_label = i
            if i == 0:
                slice_label = "high-priority"
            elif i == 1:
                slice_label = "low-priority"

            os.makedirs(f"{result_base_path}slice-{slice_label}")
            plot_slices_configs(f"slice-{slice_label}", f"{result_base_path}slice-{slice_label}/", result['slices'][i])
            for aw in AVERAGE_WINDOWS:
                plot_slice_results(f"service-{slice_label}", f"{result_base_path}slice-{slice_label}/", stats_per_slice[i], aw)
                # plot_slice_results_confidence(f"slice-{slice_label}", f"{result_base_path}slice-{slice_label}/", raw_stats_per_slice, i, aw)
            tmp = stats_per_slice[i]
            tmp['policy_name'] = result['name']
            slice_comparison_stats[i].append(stats_per_slice[i])

        merged_per_system = merge_stats_for_system_pow(result['environment_data'])
        os.makedirs(f"{result_base_path}system_pow")
        for aw in AVERAGE_WINDOWS:
            plot_slice_results(f"system",
                               f"{result_base_path}system_pow/", merged_per_system, aw, is_system_pow=True)
        plot_policy(f"{result_base_path}system_pow/", result['policy'], result['states'])

    os.makedirs(f"{EXPORTED_FILES_PATH}comparison/")
    for slice_i in range(len(slice_comparison_stats)):
        slice_label = slice_i
        if slice_i == 0:
            slice_label = "high-priority"
        elif slice_i == 1:
            slice_label = "low-priority"

        os.makedirs(f"{EXPORTED_FILES_PATH}/comparison/slice-{slice_label}")
        for aw in AVERAGE_WINDOWS:
            plot_slice_comparison(f"comparison slice-{slice_label}",
                                  f"{EXPORTED_FILES_PATH}/comparison/slice-{slice_label}/",
                                  slice_comparison_stats[slice_i], aw)


if __name__ == '__main__':
    main(sys.argv[1:])
