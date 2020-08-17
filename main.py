"""
Confront MPD policy based agent vs the average of N random policy based agent.
Then it find the best random policy and plot it.
"""
from slice_mdp import SliceMDP
from agent import Agent
from slice_simulator import SliceSimulator
import time
import plotter
import utils

SIMULATIONS = 100
SIMULATION_TIME = 1000

MDP_DISCOUNT_INCREMENT = 1

if __name__ == '__main__':
    arrivals = [0.5, 0.5]
    departures = [0.6, 0.4]
    mdp_stats = []
    random_stats = []

    time_start = time.time()

    # mdp agent
    best_mdp_costs = None
    best_mdp_processed = None
    best_mdp_policy = None

    slice_mdp = SliceMDP(arrivals, departures, 2, 1, alpha=0.5, c_lost=2, verbose=False)

    tmp_discount_factor = 0.
    while(tmp_discount_factor < 1.):
        policy = slice_mdp.run_value_iteration(0 + MDP_DISCOUNT_INCREMENT)


        for j in range(SIMULATIONS):
            slice_simulator = SliceSimulator(arrivals, departures, c_lost=2, simulation_time=SIMULATION_TIME, verbose=False)
            mdp_agent = Agent(slice_mdp.states, policy, slice_simulator)
            mdp_stats.append(mdp_agent.control_environment())

    plotter.plot_markov_chain(slice_mdp.states, slice_mdp.transition_matrix, slice_mdp.reward_matrix,
                              projectname="mdp-toy", view=False)

    mdp_costs = utils.get_mean_costs(mdp_stats)['mean'].sum()
    mdp_processed = utils.get_mean_processed_jobs(mdp_stats)['mean'].sum()
    print(f"[MDP with policy {policy}]: Total cumulative costs {mdp_costs}, total processed {mdp_processed}, "
          f"cost per processed {mdp_costs / mdp_processed}")

    # random agent
    best_random_costs = None
    best_random_processed = None
    best_random_policy = None

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

        tmp_costs = utils.get_mean_costs(random_stats_tmp)['mean'].sum()
        tmp_processed = utils.get_mean_processed_jobs(random_stats_tmp)['mean'].sum()
        print(f"[Random with policy {random_policy}]: Total cumulative costs {tmp_costs}, total processed {tmp_processed}, "
              f"cost per processed {tmp_costs / tmp_processed}")

        if best_random_costs is None or tmp_costs < best_random_costs:
            best_random_costs = tmp_costs
            best_random_processed = tmp_processed
            best_random_policy = random_policy

    print(f"Best random policy found is {best_random_policy} with costs {best_random_costs} and processed {best_random_processed}, "
          f"cost per processed {best_random_costs / best_random_processed}")

    print(f"Simulation done in {(time.time() - time_start) / 60} minutes")

    # plotting!
    utils.easy_plot("mdp-toy", mdp_stats, True)
    utils.easy_plot("random-toy", random_stats, True)


