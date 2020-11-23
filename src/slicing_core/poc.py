# SLICING_CORE MAIN

from src.slicing_core.policy import MultiSliceMdpPolicy, SingleSliceMdpPolicy
from src.slicing_core.environment import MultiSliceSimulator
from src.slicing_core.agent import NetworkOperator

import src.slicing_core.config as config
import src.slicing_core.utils as utils

import time
import logging
import os
import shutil


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


def main():
    os.makedirs(config.EXPORTED_FILES_PATH)
    shutil.copyfile(config.CONFIG_FILE_PATH, f"{config.EXPORTED_FILES_PATH}config.yaml")
    logging.basicConfig(filename=config.LOG_FILE_PATH, level=logging.INFO)

    logging.info(f"Latest commit available at {utils.get_last_commit_link()}")

    # ---- POLICY STUFF ------------------------
    policy_conf = config.PolicyConfig()

    priority_slices = [SingleSliceMdpPolicy(policy_conf, 0)]
    priority_slices[0].init()
    priority_slices[0].calculate_policy()
    print(f"Default server max cap is {policy_conf.server_max_cap}")
    policy_conf.server_max_cap -= min(priority_slices[0].policy)
    print(f"Server max cap for the second slice is now {policy_conf.server_max_cap}")
    priority_slices.append(SingleSliceMdpPolicy(policy_conf, 1))
    priority_slices[1].init()
    priority_slices[1].calculate_policy()




    policy = MultiSliceMdpPolicy(policy_conf)
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
            'transition_matrix': policy.transition_matrix,
            'reward_matrix': policy.reward_matrix,
            'environment_data': add_real_costs_to_stats(agent.history, policy_conf.slices)
        },
        config.RESULTS_FILE_PATH)

    # call the plotter script
    result_file_absolute_path = os.path.abspath(config.RESULTS_FILE_PATH).replace('\\', '/')
    os.system(f"cd ../plotter && python main.py -d {result_file_absolute_path}")


if __name__ == "__main__":
    main()