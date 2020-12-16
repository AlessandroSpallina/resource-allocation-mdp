# SLICING_CORE MAIN

from src.slicing_core.policy import MultiSliceMdpPolicy, PriorityMultiSliceMdpPolicy, MultiSliceStaticPolicy,\
    CachedPolicy
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
    USAGE = "main.py -w <workingDirectory> -c <configPath> -n <simulationName>"
    to_return = {}
    try:
        # help, config (path), name (directory name of the results)
        opts, args = getopt.getopt(argv, "hw:c:n:", ["wdir=", "config=", "name="])
    except getopt.GetoptError:
        print(USAGE)
        sys.exit(2)
    for opt, arg in opts:
        if opt == '-h':
            print(USAGE)
            sys.exit()
        elif opt in ('-w', '--wdir'):
            to_return['wdir'] = arg
        elif opt in ('-c', '--config'):
            to_return['config'] = arg
        elif opt in ('-n', '--name'):
            to_return['name'] = arg

    return to_return


def main(argv):
    # ---- CLI ARGS HANDLING -----------------------
    cli_args = cli_handler(argv)
    if 'wdir' in cli_args:
        os.chdir(cli_args['wdir'])
        print(f"changed working dir to {os.getcwd()}")
    if 'name' in cli_args:  # name influences the exported_files_path
        config.EXPORTED_FILES_PATH = f"{config.EXPORTED_FILES_PATH[:-11]}{cli_args['name']}/"
    if 'config' in cli_args:  # config influences the config_file_path
        config.CONFIG_FILE_PATH = cli_args['config']
    # ----------------------------------------------

    os.makedirs(config.EXPORTED_FILES_PATH)
    shutil.copyfile(config.CONFIG_FILE_PATH, f"{config.EXPORTED_FILES_PATH}config.yaml")
    logging.basicConfig(filename=f"{config.EXPORTED_FILES_PATH}{config.LOG_FILENAME}",
                        level=logging.INFO,
                        format="%(asctime)s::%(levelname)s::%(filename)s::%(lineno)d::%(message)s",
                        datefmt="%d%b %H:%M")

    logging.info(f"Latest commit available at {utils.get_last_commit_link()}")

    # ---- POLICY STUFF ------------------------
    policy_conf = config.PolicyConfig(custom_path=config.CONFIG_FILE_PATH)
    policies = [
            CachedPolicy(policy_conf, PriorityMultiSliceMdpPolicy),
            CachedPolicy(policy_conf, MultiSliceStaticPolicy)
        ]
    start_time = time.time()

    for policy in policies:
        policy.init()

    logging.info(f"Initialization done in {time.time() - start_time} seconds")
    start_time = time.time()

    for policy in policies:
        policy.calculate_policy()

    logging.info(f"Policy calculation done in {time.time() - start_time} seconds")
    # ------------------------------------------

    # ---- ENVIRONMENT & AGENT STUFF --------------------
    simulation_conf = config.SimulationConfig()
    start_time = time.time()
    agents = [NetworkOperatorSimulator(policy, simulation_conf) for policy in policies]

    for agent in agents:
        agent.start_automatic_control()
    # ---------------------------------------------------

    logging.info(f"Simulation done in {time.time() - start_time} seconds")

    utils.export_data(
        [
            {
                'name': policies[i].obj.__class__.__name__,
                'policy': policies[i].policy,
                'states': policies[i].states,
                'slices': [
                    {
                        'arrivals_histogram': policy_conf.slices[j].arrivals_histogram,
                        'server_capacity_histogram': policy_conf.slices[j].server_capacity_histogram,
                    }
                    for j in range(policy_conf.slice_count)
                ],
                'environment_data': agents[i].history
            } for i in range(len(policies))
        ],
        f"{config.EXPORTED_FILES_PATH}{config.RESULTS_FILENAME}")

    # call the plotter script
    result_file_absolute_path = \
        os.path.abspath(f"{config.EXPORTED_FILES_PATH}{config.RESULTS_FILENAME}").replace('\\', '/')
    os.system(f"cd ../../ && python -m src.plotter.main -d {result_file_absolute_path} -w ./src/plotter/")


if __name__ == "__main__":
    main(sys.argv[1:])
