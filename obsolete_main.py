"""
Confront MPD policy based agent (discount 0.8) vs the average of N random policy based agent.
"""
from slice_mdp import SliceMDP
from agent import Agent
from slice_simulator import SliceSimulator
import time
import plotter
import utils

SIMULATIONS = 100
SIMULATION_TIME = 1000

if __name__ == '__main__':
    arrivals = [0.5, 0.5]
    departures = [0.6, 0.4]
    mdp_stats = []
    random_stats = []

    time_start = time.time()

    # mdp agent
    slice_mdp = SliceMDP(arrivals, departures, 2, 1, alpha=0.5, c_lost=2, verbose=False)
    policy = slice_mdp.run_value_iteration(0.8)
    plotter.plot_markov_chain(slice_mdp.states, slice_mdp.transition_matrix, slice_mdp.reward_matrix,
                              projectname="mdp-toy", view=False)

    for i in range(SIMULATIONS):
        slice_simulator = SliceSimulator(arrivals, departures, c_lost=2, simulation_time=SIMULATION_TIME, verbose=False)
        mdp_agent = Agent(slice_mdp.states, policy, slice_simulator)
        mdp_stats.append(mdp_agent.control_environment())

    # random agent
    for i in range(SIMULATIONS):
        random_stats_tmp = []
        random_policy = utils.generate_random_policy(len(slice_mdp.states), 3)

        for j in range(SIMULATIONS):
            random_simulation = SliceSimulator(arrivals, departures, c_lost=2, simulation_time=SIMULATION_TIME, verbose=False)
            random_agent = Agent(slice_mdp.states, random_policy, random_simulation)
            random_stats_tmp.append(random_agent.control_environment())

        random_stats.append({'costs_per_timeslot': utils.get_mean_costs(random_stats_tmp)['mean'].tolist(),
                             'processed_jobs_per_timeslot': utils.get_mean_processed_jobs(random_stats_tmp)['mean'].tolist(),
                             'lost_jobs_per_timeslot': utils.get_mean_lost_jobs(random_stats_tmp)['mean'].tolist(),
                             'wait_time_per_job': utils.get_mean_wait_time(random_stats_tmp)['mean'].tolist()})

    print(f"Simulation done in {(time.time() - time_start) / 60} minutes")

    # plotting!
    utils.easy_plot("mdp-toy", mdp_stats, True)
    utils.easy_plot("random-toy", random_stats, True)

