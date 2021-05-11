# SLICING_CORE MAIN

from src.slicing_core.policy import MultiSliceMdpPolicy, PriorityMultiSliceMdpPolicy, MultiSliceStaticPolicy,\
    CachedPolicy, SimplifiedPriorityMultiSliceMdpPolicy, SequentialPriorityMultiSliceMdpPolicy
from src.slicing_core.agent import NetworkOperatorSimulator

import src.slicing_core.config as config
import src.slicing_core.utils as utils

import time
import logging
import os
import shutil
import sys
import getopt
import numpy as np
import multiprocessing


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


def background_run_plotting(result_file_absolute_path):
    os.system(f"cd ../../ && python -m src.plotter.main -d {result_file_absolute_path} -w ./src/plotter/")


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
    confs = [
        config.MdpPolicyConfig(custom_path=config.CONFIG_FILE_PATH),
        config.MdpPolicyConfig(custom_path=config.CONFIG_FILE_PATH),
        config.StaticPolicyConfig(custom_path=config.CONFIG_FILE_PATH),
    ]

    policies = [
            CachedPolicy(confs[0], SequentialPriorityMultiSliceMdpPolicy),
            CachedPolicy(confs[1], SimplifiedPriorityMultiSliceMdpPolicy),
            CachedPolicy(confs[2], MultiSliceStaticPolicy),
        ]

    start_time = time.time()

    for policy in policies:
        policy.init()

    logging.info(f"Initialization done in {time.time() - start_time} seconds")
    start_time = time.time()

    for policy in policies:
        policy.calculate_policy()

    # the last policy is a static that allocates the maximum required by the mdp policy (so, the optimal static alloc)
    priority_static_conf = config.StaticPolicyConfig(custom_path=config.CONFIG_FILE_PATH)
    priority_static_conf.set_allocation(0, int(np.array(policies[0].policy)[:, 0].max()))
    priority_static_conf.set_allocation(1, int(np.array(policies[0].policy)[:, 1].max()))

    priority_static = CachedPolicy(priority_static_conf, MultiSliceStaticPolicy)
    priority_static.init()
    priority_static.calculate_policy()

    confs.append(priority_static_conf)
    policies.append(priority_static)

    logging.info(f"Policy calculation done in {time.time() - start_time} seconds")
    # ------------------------------------------

    # ---- ENVIRONMENT & AGENT STUFF --------------------
    simulation_conf = config.SimulatorConfig(custom_path=config.CONFIG_FILE_PATH)
    start_time = time.time()
    agents = [NetworkOperatorSimulator(policy, simulation_conf) for policy in policies]

    for agent in agents:
        agent.start_automatic_control()
    # ---------------------------------------------------

    logging.info(f"Simulation done in {time.time() - start_time} seconds")

    utils.serialize_data(
        [
            {
                'name': f"{i}_{policies[i].obj.__class__.__name__}",
                'policy': policies[i].policy,
                'states': policies[i].json_states,
                'slices': [
                    {
                        'arrivals_histogram': confs[i].slices[j].arrivals_histogram,
                        'server_capacity_histogram': confs[i].slices[j].server_capacity_histogram,
                    }
                    for j in range(confs[i].slice_count)
                ],
                'environment_data': agents[i].history,
                'environment_data_std': agents[i].history_std,
                # 'environment_data_raw': agents[i].history_raw
            } for i in range(len(policies))
        ],
        f"{config.EXPORTED_FILES_PATH}{config.RESULTS_FILENAME}")

    # call the plotter script
    result_file_absolute_path = \
        os.path.abspath(f"{config.EXPORTED_FILES_PATH}{config.RESULTS_FILENAME}").replace('\\', '/')
    print(f"cd ../../ && python -m src.plotter.main -d {result_file_absolute_path} -w ./src/plotter/")
    multiprocessing.Process(target=background_run_plotting, args=(result_file_absolute_path,)).start()


if __name__ == "__main__":
    main(sys.argv[1:])
