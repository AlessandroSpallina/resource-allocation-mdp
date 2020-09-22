"""
Simulate and plot different mdp policy (different gamma) and other policies like conservative or smart conservative
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


def main():
    if not os.path.exists(STORAGE_PATH):
        os.makedirs(STORAGE_PATH)
    logging.basicConfig(filename=f"{STORAGE_PATH}report.log", level=logging.INFO)

    logging.info(f"Latest commit available at {utils.get_last_commit_link()}")

    shutil.copyfile("./config.yaml", f"{STORAGE_PATH}config.yaml")
    os.chdir(STORAGE_PATH)

    conf = utils.read_config(True)

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

    MAX_POINTS_IN_PLOT = conf['max_points_in_plot']

    stats = {}

    time_start = time.time()

    # policy generations
    slice_mdp = SliceMDP(ARRIVALS, DEPARTURES, QUEUE_SIZE, SERVER_MAX_CAP, alpha=ALPHA, beta=BETA, gamma=GAMMA,
                         c_server=C_SERVER, c_job=C_JOB, c_lost=C_LOST, algorithm=MDP_ALGORITHM,
                         periods=SIMULATION_TIME, delayed_action=DELAYED_ACTION, verbose=False)

    policies = slice_mdp.run([(i / 10) - 1e-10 for i in range(round(DISCOUNT_START_VALUE * 10),
                                                              round(DISCOUNT_END_VALUE * 10) + 1,
                                                              round(MDP_DISCOUNT_INCREMENT * 10))])

    policies['all-on'] = utils.generate_all_on_policy(len(slice_mdp.states))
    policies['conservative'] = utils.generate_conservative_policy(slice_mdp.states)

    logging.info(f"*** Generated {len(policies)} policies in {(time.time() - time_start) / 60} minutes ***")
    time_start = time.time()

    # simulations with generated policies
    for i in policies:
        stats_tmp = []
        for j in range(SIMULATIONS):
            simulator = SliceSimulator(ARRIVALS, DEPARTURES, queue_size=QUEUE_SIZE, max_server_num=SERVER_MAX_CAP,
                                       alpha=ALPHA, beta=BETA, gamma=GAMMA, c_server=C_SERVER, c_job=C_JOB,
                                       c_lost=C_LOST, simulation_time=SIMULATION_TIME, delayed_action=DELAYED_ACTION,
                                       verbose=False)
            agent = Agent(slice_mdp.states, policies[i], simulator)
            stats_tmp.append(agent.control_environment())
        stats[i] = {'costs_per_timeslot': utils.get_mean_costs(stats_tmp)['mean'],
                    'component_costs_per_timeslot': utils.get_mean_component_costs(stats_tmp)['mean'],
                    'processed_jobs_per_timeslot': utils.get_mean_processed_jobs(stats_tmp)['mean'],
                    'lost_jobs_per_timeslot': utils.get_mean_lost_jobs(stats_tmp)['mean'],
                    'wait_time_in_the_queue_per_job': utils.get_mean_wait_time_in_the_queue(stats_tmp)['mean'],
                    'wait_time_in_the_system_per_job': utils.get_mean_wait_time_in_the_system(stats_tmp)['mean'],
                    'jobs_in_queue_per_timeslot': utils.get_mean_jobs_in_queue(stats_tmp)['mean'],
                    'active_servers_per_timeslot': utils.get_mean_active_servers(stats_tmp)['mean'],
                    'policy': utils.get_matrix_policy(policies[i], SERVER_MAX_CAP)}

    logging.info(f"*** Simulation done in {(time.time() - time_start) / 60} minutes ***")
    time_start = time.time()

    # shutil.copyfile("./config.yaml", f"{STORAGE_PATH}config.yaml")
    # os.chdir(STORAGE_PATH)

    # plot generation and export on filesystem
    # plotter.plot_markov_chain(slice_mdp.states, slice_mdp.transition_matrix, slice_mdp.reward_matrix,
    #                           projectname="mdp-agent", view=False)

    for i in stats:
        utils.easy_plot(i, stats[i], MAX_POINTS_IN_PLOT)

    plotter.bar(ydata={"arrivals": ARRIVALS}, projectname="common", title="Arrivals Histogram",
                xlabel="job", ylabel="arrival probability")
    plotter.bar(ydata={"departures": DEPARTURES}, projectname="common",
                title="Server Capacity Histogram (Departures Histogram)", xlabel="job", ylabel="departure probability")

    utils.comparison_plot("common", stats, MAX_POINTS_IN_PLOT)

    logging.info(f"*** Plotting done in {(time.time() - time_start) / 60} minutes ***")


if __name__ == '__main__':
    main()
