"""
"""
import shutil
import time
import logging

import utils
import plotter
from agent import Agent
from slice_mdp import UnitaryAllocationSliceMDP, MultipleAllocationSliceMDP
from multi_slice_mdp import MultiSliceMDP
from multi_slice_simulator import MultiSliceSimulator
from slice_simulator import SliceSimulator
import os
import getopt
import sys


def cli_handler(argv):
    USAGE = "multislice_main.py -c <configPath> -n <simulationName>"
    to_return = {}
    try:
        # help, config (path), name (directory name of the results)
        opts, args = getopt.getopt(argv, "hc:n:", ["config=", "name="])
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
        elif opt in ('-c', '--config'):
            to_return['config'] = arg
        elif opt in ('-n', '--name'):
            to_return['name'] = arg

    return to_return


def main(argv):
    cli_args = cli_handler(argv)
    STORAGE_PATH = f"../../res/exported/{cli_args['name'] if 'name' in cli_args else int(time.time())}/"
    CONFIG_PATH = cli_args['config'] if 'config' in cli_args else "config.yaml"

    if not os.path.exists(STORAGE_PATH):
        os.makedirs(STORAGE_PATH)
    logging.basicConfig(filename=f"{STORAGE_PATH}report.log", level=logging.INFO)

    logging.info(f"Latest commit available at {utils.get_last_commit_link()}")

    shutil.copyfile(CONFIG_PATH, f"{STORAGE_PATH}config.yaml")
    os.chdir(STORAGE_PATH)

    conf = utils.read_config(path=CONFIG_PATH, verbose=True)

    ARRIVALS = conf['arrivals_histogram']
    DEPARTURES = conf['departures_histogram']
    QUEUE_SIZE = conf['queue_size']
    SERVER_MAX_CAP = conf['server_max_cap']

    # normalize alpha, beta and gamma
    ALPHA = conf['alpha'] / (conf['alpha'] + conf['beta'] + conf['gamma'])
    BETA = conf['beta'] / (conf['alpha'] + conf['beta'] + conf['gamma'])
    GAMMA = conf['gamma'] / (conf['alpha'] + conf['beta'] + conf['gamma'])
    logging.info(f"Normalized alpha {ALPHA}, beta {BETA} and gamma {GAMMA}")

    C_SERVER = conf['c_server']
    C_JOB = conf['c_job']
    C_LOST = conf['c_lost']

    SIMULATIONS = conf['simulations']
    SIMULATION_TIME = conf['simulation_time']
    MDP_DISCOUNT_INCREMENT = conf['mdp_discount_increment']
    DISCOUNT_START_VALUE = conf['mdp_discount_start_value']
    DISCOUNT_END_VALUE = conf['mdp_discount_end_value']
    MDP_ALGORITHM = conf['mdp_algorithm']

    DELAYED_ACTION = conf['delayed_action']

    AVERAGE_WINDOW_IN_PLOT = conf['average_window_in_plot']

    ARRIVAL_PROCESSING_PHASES = conf['arrival_processing_phase']

    stats = {}

    time_start = time.time()

    slices = [
        MultipleAllocationSliceMDP(ARRIVALS, DEPARTURES, QUEUE_SIZE, 3, alpha=ALPHA, beta=BETA,
                                   gamma=GAMMA, c_server=C_SERVER, c_job=C_JOB, c_lost=C_LOST,
                                   algorithm=MDP_ALGORITHM, periods=SIMULATION_TIME,
                                   delayed_action=False, label="ma-0",
                                   arrival_processing_phase=True, verbose=False),
        MultipleAllocationSliceMDP([0., 0.6, 0.4], DEPARTURES, 5, 3, alpha=ALPHA, beta=BETA,
                                   gamma=GAMMA, c_server=C_SERVER, c_job=C_JOB, c_lost=C_LOST,
                                   algorithm=MDP_ALGORITHM, periods=SIMULATION_TIME,
                                   delayed_action=False, label="ma-1",
                                   arrival_processing_phase=True, verbose=False)
    ]
    multi = MultiSliceMDP(slices)

    multi_policies = multi.run([(i / 10) for i in range(round(DISCOUNT_START_VALUE * 10),
                                                        round(DISCOUNT_END_VALUE * 10) + 1,
                                                        round(MDP_DISCOUNT_INCREMENT * 10))])


    # -----------------------

    policies = {**multi_policies}

    logging.info(f"*** Generated {len(policies)} policies in {(time.time() - time_start) / 60} minutes ***")
    time_start = time.time()

    simulators = [
        SliceSimulator(ARRIVALS, DEPARTURES, queue_size=QUEUE_SIZE, max_server_num=3,
                       alpha=ALPHA, beta=BETA, gamma=GAMMA, c_server=C_SERVER, c_job=C_JOB,
                       c_lost=C_LOST, simulation_time=SIMULATION_TIME, delayed_action=False,
                       arrival_processing_phase=ARRIVAL_PROCESSING_PHASES, verbose=False),
        SliceSimulator([0., 0.6, 0.4], DEPARTURES, queue_size=5, max_server_num=3,
                       alpha=ALPHA, beta=BETA, gamma=GAMMA, c_server=C_SERVER, c_job=C_JOB,
                       c_lost=C_LOST, simulation_time=SIMULATION_TIME, delayed_action=False,
                       arrival_processing_phase=ARRIVAL_PROCESSING_PHASES, verbose=False)

        ]

    # simulations with generated policies
    for i in policies:
        stats_tmp = []
        for j in range(SIMULATIONS):
            simulator = MultiSliceSimulator(simulators, multi.actions)
            agent = Agent(multi.states, policies[i], simulator)
            stats_tmp.append(agent.control_environment()[0])
            stats_tmp.append(agent.control_environment()[1])
        print("qui")
        stats[i+"0"] = {'costs_per_timeslot': stats_tmp[0]['costs_per_timeslot'],
                        'component_costs_per_timeslot': stats_tmp[0]['component_costs_per_timeslot'],
                        'processed_jobs_per_timeslot': stats_tmp[0]['processed_jobs_per_timeslot'],
                        'lost_jobs_per_timeslot': stats_tmp[0]['lost_jobs_per_timeslot'],
                        # 'wait_time_in_the_queue_per_job': utils.get_mean_wait_time_in_the_queue(stats_tmp)[0]['mean'],
                        # 'wait_time_in_the_system_per_job': utils.get_mean_wait_time_in_the_system(stats_tmp)[0]['mean'],
                        # 'jobs_in_queue_per_timeslot': utils.get_mean_jobs_in_queue(stats_tmp[0])['mean'],
                        # 'active_servers_per_timeslot': utils.get_mean_active_servers(stats_tmp[0])['mean'],
                        'policy': utils.get_matrix_policy(policies[i], SERVER_MAX_CAP)}
        stats[i + "1"] = {'costs_per_timeslot': stats_tmp[1]['costs_per_timeslot'],
                        'component_costs_per_timeslot': stats_tmp[1]['component_costs_per_timeslot'],
                        'processed_jobs_per_timeslot': stats_tmp[1]['processed_jobs_per_timeslot'],
                        'lost_jobs_per_timeslot': stats_tmp[1]['lost_jobs_per_timeslot'],
                          # 'wait_time_in_the_queue_per_job': utils.get_mean_wait_time_in_the_queue(stats_tmp)[1]['mean'],
                          # 'wait_time_in_the_system_per_job': utils.get_mean_wait_time_in_the_system(stats_tmp)[1]['mean'],
                          # 'jobs_in_queue_per_timeslot': utils.get_mean_jobs_in_queue(stats_tmp[1])['mean'],
                          # 'active_servers_per_timeslot': utils.get_mean_active_servers(stats_tmp[1])['mean'],
                          'policy': utils.get_matrix_policy(policies[i], SERVER_MAX_CAP)}

    logging.info(f"*** Simulation done in {(time.time() - time_start) / 60} minutes ***")
    time_start = time.time()

    for i in stats:
        utils.easy_plot(i, stats[i], AVERAGE_WINDOW_IN_PLOT)

    plotter.bar(ydata={"arrivals": ARRIVALS}, projectname="common", title="Arrivals Histogram",
                xlabel="job", ylabel="arrival probability")
    plotter.bar(ydata={"departures": DEPARTURES}, projectname="common",
                title="Server Capacity Histogram (Departures Histogram)", xlabel="job", ylabel="departure probability")

    # utils.comparison_plot("common", stats, AVERAGE_WINDOW_IN_PLOT)

    logging.info(f"*** Plotting done in {(time.time() - time_start) / 60} minutes ***")

    # -----------------------


if __name__ == '__main__':
    main(sys.argv[1:])
