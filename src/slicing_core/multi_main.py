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

    # -------------------
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

    print(multi.run(1))

    # -----------------------


if __name__ == '__main__':
    main(sys.argv[1:])
