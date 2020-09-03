"""
Find the best discount factor and the best random policy
"""
import os
import shutil
import time
import logging

import plotter
import utils
from agent import Agent
from slice_mdp import SliceMDP
from slice_simulator import SliceSimulator

STORAGE_PATH = "./res/exported/{}/".format(int(time.time()))

if __name__ == '__main__':
    if not os.path.exists(STORAGE_PATH):
        os.makedirs(STORAGE_PATH)
    logging.basicConfig(filename=f"{STORAGE_PATH}report.log", level=logging.INFO)

    logging.info(f"Latest commit available at {utils.get_last_commit_link()}")

    conf = utils.read_config(True)

    ARRIVALS = conf['arrivals_histogram']
    DEPARTURES = conf['departures_histogram']
    QUEUE_SIZE = conf['queue_size']
    SERVER_MAX_CAP = conf['server_max_cap']
    ALPHA = conf['alpha']
    C_SERVER = conf['c_server']
    C_LOST = conf['c_lost']

    SIMULATIONS = conf['simulations']
    SIMULATION_TIME = conf['simulation_time']
    RANDOM_POLICY_ATTEMPT = conf['random_policy_attempts']
    MDP_DISCOUNT_INCREMENT = conf['mdp_discount_increment']

    tmp_discount_factor = conf['mdp_discount_start_value']

    mdp_stats = []
    random_stats = []

    time_start = time.time()

    # mdp agent
    best_mdp_costs = None
    best_mdp_processed = None
    best_mdp_lost = None
    best_mdp_policy = None
    best_discount_factor = None

    slice_mdp = SliceMDP(ARRIVALS, DEPARTURES, QUEUE_SIZE, SERVER_MAX_CAP, alpha=ALPHA, c_server=C_SERVER, c_lost=C_LOST, verbose=False)

    while tmp_discount_factor <= 1.:
        mdp_stats_tmp = []
        policy = slice_mdp.run_value_iteration(tmp_discount_factor)

        for j in range(SIMULATIONS):
            slice_simulator = SliceSimulator(ARRIVALS, DEPARTURES, queue_size=QUEUE_SIZE, max_server_num=SERVER_MAX_CAP,
                                             alpha=ALPHA, c_server=C_SERVER, c_lost=C_LOST,
                                             simulation_time=SIMULATION_TIME, verbose=False)
            mdp_agent = Agent(slice_mdp.states, policy, slice_simulator)
            mdp_stats_tmp.append(mdp_agent.control_environment())

        tmp_costs = utils.get_mean_costs(mdp_stats_tmp)['mean'].sum()
        tmp_processed = utils.get_mean_processed_jobs(mdp_stats_tmp)['mean'].sum()
        tmp_lost = utils.get_mean_lost_jobs(mdp_stats_tmp)['mean'].sum()

        logging.info(f"[MDP with policy {policy} (discount {tmp_discount_factor})]: Total cumulative costs {tmp_costs}, "
                     f"total processed {tmp_processed}, total lost jobs {tmp_lost},"
                     f"cost per processed {tmp_costs / tmp_processed}")

        if best_mdp_costs is None or tmp_costs < best_mdp_costs:
            best_mdp_costs = tmp_costs
            best_mdp_processed = tmp_processed
            best_mdp_lost = tmp_lost
            best_mdp_policy = policy
            best_discount_factor = tmp_discount_factor
            # N.B. random_stats contain the stats of the best policy simulated!
            mdp_stats = {'costs_per_timeslot': utils.get_mean_costs(mdp_stats_tmp)['mean'],
                         'processed_jobs_per_timeslot': utils.get_mean_processed_jobs(mdp_stats_tmp)['mean'],
                         'lost_jobs_per_timeslot': utils.get_mean_lost_jobs(mdp_stats_tmp)['mean'],
                         'wait_time_per_job': utils.get_mean_wait_time(mdp_stats_tmp)['mean']}

        tmp_discount_factor += MDP_DISCOUNT_INCREMENT

    # random agent
    best_random_costs = None
    best_random_processed = None
    best_random_lost = None
    best_random_policy = None

    for i in range(RANDOM_POLICY_ATTEMPT):
        random_stats_tmp = []
        random_policy = utils.generate_random_policy(len(slice_mdp.states), 3)

        for j in range(SIMULATIONS):
            random_simulation = SliceSimulator(ARRIVALS, DEPARTURES, queue_size=QUEUE_SIZE,
                                               max_server_num=SERVER_MAX_CAP, alpha=ALPHA, c_server=C_SERVER,
                                               c_lost=C_LOST, simulation_time=SIMULATION_TIME, verbose=False)
            random_agent = Agent(slice_mdp.states, random_policy, random_simulation)
            random_stats_tmp.append(random_agent.control_environment())

        tmp_costs = utils.get_mean_costs(random_stats_tmp)['mean'].sum()
        tmp_processed = utils.get_mean_processed_jobs(random_stats_tmp)['mean'].sum()
        tmp_lost = utils.get_mean_lost_jobs(random_stats_tmp)['mean'].sum()

        logging.info(f"[Random with policy {random_policy}]: Total cumulative costs {tmp_costs}, "
                     f"total processed {tmp_processed}, total lost jobs {tmp_lost}"
                     f"cost per processed {tmp_costs / tmp_processed}")

        if best_random_costs is None or tmp_costs < best_random_costs:
            best_random_costs = tmp_costs
            best_random_processed = tmp_processed
            best_random_lost = tmp_lost
            best_random_policy = random_policy
            # N.B. random_stats contain the stats of the best policy simulated!
            random_stats = {'costs_per_timeslot': utils.get_mean_costs(random_stats_tmp)['mean'],
                            'processed_jobs_per_timeslot': utils.get_mean_processed_jobs(random_stats_tmp)['mean'],
                            'lost_jobs_per_timeslot': utils.get_mean_lost_jobs(random_stats_tmp)['mean'],
                            'wait_time_per_job': utils.get_mean_wait_time(random_stats_tmp)['mean']}
            # print(f"BEST RANDOM HERE {random_stats['wait_time_per_job']}")

    logging.info(f"* Best mdp policy found is {best_mdp_policy} with costs {best_mdp_costs} "
                 f"and processed {best_mdp_processed} and lost jobs {best_mdp_lost} "
                 f"cost per processed {best_mdp_costs / best_mdp_processed}")

    logging.info(f"* Best random policy found is {best_random_policy} with costs {best_random_costs} "
                 f"and processed {best_random_processed} and lost jobs {best_random_lost}, "
                 f"cost per processed {best_random_costs / best_random_processed}")

    logging.info(f"*** Simulation done in {(time.time() - time_start) / 60} minutes ***")

    shutil.copyfile("./config.yaml", f"{STORAGE_PATH}config.yaml")
    os.chdir(STORAGE_PATH)

    # plot generation and export on file system
    plotter.plot_markov_chain(slice_mdp.states, slice_mdp.transition_matrix, slice_mdp.reward_matrix,
                              projectname="mdp-toy", view=False)
    # utils.easy_plot("mdp-toy", "Policy {}".format(best_mdp_policy), mdp_stats, False)
    # utils.easy_plot("random-toy", "Policy {}".format(best_random_policy), random_stats, False)
    utils.easy_plot("mdp-toy", "", mdp_stats, False)
    utils.easy_plot("random-toy", "", random_stats, False)
