# SLICING_CORE MAIN

from src.slicing_core.policy import MultiSliceMdpPolicy
from src.slicing_core.environment import MultiSliceSimulator
from src.slicing_core.agent import NetworkOperator

import src.slicing_core.config as config
import src.slicing_core.utils as utils

import time
import logging
import os


def main():
    os.makedirs(config.EXPORTED_FILES_PATH)
    logging.basicConfig(filename=config.LOG_FILE_PATH, level=logging.INFO)

    logging.info(f"Latest commit available at {utils.get_last_commit_link()}")

    policy_conf = config.PolicyConfig()
    policy = MultiSliceMdpPolicy(policy_conf)

    start_time = time.time()

    policy.init()

    logging.info(f"Initialization done in {time.time() - start_time} seconds")
    start_time = time.time()

    policy.calculate_policy()

    logging.info(f"Policy calculation done in {time.time() - start_time} seconds")

    environment_conf = config.EnvironmentConfig()
    environment = MultiSliceSimulator(environment_conf)

    start_time = time.time()

    agent = NetworkOperator(policy, environment, policy_conf.timeslots)
    agent.start_automatic_control()

    logging.info(f"Simulation done in {time.time() - start_time} seconds")

    utils.export_data({
        'policy': policy.policy,
        'transition_matrix': policy.transition_matrix,
        'reward_matrix': policy.reward_matrix,
        'simulation_data': agent.history}, config.RESULTS_FILE_PATH)

    # call the plotter script
    result_file_absolute_path = os.path.abspath(config.RESULTS_FILE_PATH).replace('\\', '/')
    os.system(f"cd ../plotter && python main.py -d {result_file_absolute_path}")


if __name__ == "__main__":
    main()
