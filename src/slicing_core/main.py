# SLICING_CORE MAIN

from src.slicing_core.policy import MultiSliceMdpPolicy, PriorityMultiSliceMdpPolicy, CachedPolicy
from src.slicing_core.environment import MultiSliceSimulator
from src.slicing_core.agent import NetworkOperatorSimulator

import src.slicing_core.config as config
import src.slicing_core.utils as utils

import time
import logging
import os
import shutil
import sys
import getopt


def cli_handler(argv):
    USAGE = "main.py -w <workingDirectory>"
    to_return = {}
    try:
        # help, config (path), name (directory name of the results)
        opts, args = getopt.getopt(argv, "hw:", ["wdir="])
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
        elif opt in ('-w', '--wdir'):
            to_return['wdir'] = arg

    return to_return


def main(argv):
    cli_args = cli_handler(argv)
    if 'wdir' in cli_args:
        os.chdir(cli_args['wdir'])
        print(f"changed working dir to {os.getcwd()}")

    os.makedirs(config.EXPORTED_FILES_PATH)
    shutil.copyfile(config.CONFIG_FILE_PATH, f"{config.EXPORTED_FILES_PATH}config.yaml")
    logging.basicConfig(filename=config.LOG_FILE_PATH, level=logging.INFO)

    logging.info(f"Latest commit available at {utils.get_last_commit_link()}")

    # ---- POLICY STUFF ------------------------
    policy_conf = config.PolicyConfig()
    policy = CachedPolicy(policy_conf, PriorityMultiSliceMdpPolicy)
    start_time = time.time()
    policy.init()
    logging.info(f"Initialization done in {time.time() - start_time} seconds")
    start_time = time.time()
    policy.calculate_policy()
    logging.info(f"Policy calculation done in {time.time() - start_time} seconds")
    # ------------------------------------------

    # ---- ENVIRONMENT & AGENT STUFF --------------------
    simulation_conf = config.SimulationConfig()
    start_time = time.time()
    agent = NetworkOperatorSimulator(policy, simulation_conf)
    agent.start_automatic_control()
    # ---------------------------------------------------

    logging.info(f"Simulation done in {time.time() - start_time} seconds")

    utils.export_data(
        {
            'policy': policy.policy,
            'environment_data': agent.history
        },
        config.RESULTS_FILE_PATH)

    # call the plotter script
    result_file_absolute_path = os.path.abspath(config.RESULTS_FILE_PATH).replace('\\', '/')
    os.system(f"cd ../plotter && python main.py -d {result_file_absolute_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
