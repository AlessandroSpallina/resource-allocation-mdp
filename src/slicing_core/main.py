# SLICING_CORE MAIN

from src.slicing_core.policy import MultiSliceMdpPolicy, CachedPolicy
from src.slicing_core.environment import MultiSliceSimulator
from src.slicing_core.agent import NetworkOperator

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
        elif opt in ('-c', '--wdir'):
            to_return['wdir'] = arg

    return to_return


def add_real_costs_to_stats(environment_history, slices_paramethers):
    # C = alpha * C_k * num of jobs in the queue + beta * C_n * num of server + gamma * C_l * num of lost jobs
    to_return = []
    for ts in environment_history:
        ts_tmp = []
        multislice_states = ts['state']
        lost_jobs = ts['lost_jobs']
        for i in range(len(slices_paramethers)):
            cost1 = slices_paramethers[i].alpha * slices_paramethers[i].c_job * multislice_states[i].k
            cost2 = slices_paramethers[i].beta * slices_paramethers[i].c_server * multislice_states[i].n
            cost3 = slices_paramethers[i].gamma * slices_paramethers[i].c_lost * lost_jobs[i]
            ts_tmp.append(cost1 + cost2 + cost3)
        to_return.append(ts)
        to_return[-1]['cost'] = ts_tmp
    return to_return


def main(argv):
    cli_args = cli_handler(argv)
    if 'wdir' in cli_args:
        os.chdir(cli_args['wdir'])

    os.makedirs(config.EXPORTED_FILES_PATH)
    shutil.copyfile(config.CONFIG_FILE_PATH, f"{config.EXPORTED_FILES_PATH}config.yaml")
    logging.basicConfig(filename=config.LOG_FILE_PATH, level=logging.INFO)

    logging.info(f"Latest commit available at {utils.get_last_commit_link()}")

    # ---- POLICY STUFF ------------------------
    policy_conf = config.PolicyConfig()
    policy = CachedPolicy(policy_conf, MultiSliceMdpPolicy)
    start_time = time.time()
    policy.init()
    logging.info(f"Initialization (trans&reward matrices) done in {time.time() - start_time} seconds")
    start_time = time.time()
    policy.calculate_policy()
    logging.info(f"Policy calculation done in {time.time() - start_time} seconds")
    # ------------------------------------------

    # ---- ENVIRONMENT & AGENT STUFF --------------------
    environment_conf = config.EnvironmentConfig()
    environment = MultiSliceSimulator(environment_conf)
    start_time = time.time()
    agent = NetworkOperator(policy, environment, policy_conf.timeslots)
    agent.start_automatic_control()
    # ---------------------------------------------------

    logging.info(f"Simulation done in {time.time() - start_time} seconds")

    utils.export_data(
        {
            'policy': policy.policy,
            'transition_matrix': policy.obj.transition_matrix,
            'reward_matrix': policy.obj.reward_matrix,
            'environment_data': add_real_costs_to_stats(agent.history, policy_conf.slices)
        },
        config.RESULTS_FILE_PATH)

    # call the plotter script
    result_file_absolute_path = os.path.abspath(config.RESULTS_FILE_PATH).replace('\\', '/')
    os.system(f"cd ../plotter && python main.py -d {result_file_absolute_path}")


if __name__ == "__main__":
    main(sys.argv[1:])
