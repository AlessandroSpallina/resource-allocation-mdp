from slice_mdp import SliceMDP
from agent import Agent
from slice_simulator import SliceSimulator
import plotter
import numpy as np
import random
import time

SIMULATIONS = 100


def get_mean_costs(raw_stats):
    cps = np.array([d['costs_per_timeslot'] for d in raw_stats])
    return {
        "mean": cps.mean(axis=0),
        "var": cps.var(axis=0)
        }


def get_mean_processed_jobs(raw_stats):
    pj = np.array([d['processed_jobs_per_timeslot'] for d in raw_stats])
    return {
        "mean": pj.mean(axis=0),
        "var": pj.var(axis=0)
    }


def get_mean_lost_jobs(raw_stats):
    lj = np.array([d['lost_jobs_per_timeslot'] for d in raw_stats])
    return {
        "mean": lj.mean(axis=0),
        "var": lj.var(axis=0)
    }


# 3 azioni (0,1,2) e 6 stati
def generate_random_policy(states_num, action_num):
    rpolicy = []
    for s in range(states_num):
        rpolicy.append(random.randint(0, action_num - 1))
    return tuple(rpolicy)


def easy_plot(projectname, stats, view):
    mean_costs = get_mean_costs(stats)
    mean_processed_jobs = get_mean_processed_jobs(stats)
    mean_lost_jobs = get_mean_lost_jobs(stats)

    # plotting of one simulation
    plotter.plot_cumulative({"costs": stats[0]['costs_per_timeslot'],
                             "processed jobs": stats[0]['processed_jobs_per_timeslot'],
                             "lost jobs": stats[0]['lost_jobs_per_timeslot']},
                            xlabel="timeslot", title=f"[{projectname}] One Sim Cumulative",
                            projectname=projectname, view=view)
    plotter.plot({"costs": stats[0]['costs_per_timeslot'],
                  "processed jobs": stats[0]['processed_jobs_per_timeslot'],
                  "lost jobs": stats[0]['lost_jobs_per_timeslot']},
                 xlabel="timeslot", title=f"[{projectname}] One Sim per Timeslot",
                 projectname=projectname, view=view)

    # plotting of the mean of N simulations
    plotter.plot_cumulative({"costs": mean_costs['mean'],
                             "processed jobs": mean_processed_jobs['mean'],
                             "lost jobs": mean_lost_jobs['mean']},
                            xlabel="timeslot", title=f"[{projectname}] Mean Cumulative",
                            projectname=projectname, view=view)
    plotter.plot({"costs per ts": mean_costs['mean'],
                  "processed jobs per ts": mean_processed_jobs['mean'],
                  "lost jobs per ts": mean_lost_jobs['mean']},
                 xlabel="timeslot", title=f"[{projectname}] Mean per Timeslot",
                 projectname=projectname, view=view)


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
        slice_simulator = SliceSimulator(arrivals, departures, c_lost=2, simulation_time=1000, verbose=False)
        mdp_agent = Agent(slice_mdp.states, policy, slice_simulator)
        mdp_stats.append(mdp_agent.control_environment())

    # random agent
    for i in range(SIMULATIONS):
        random_stats_tmp = []
        random_policy = generate_random_policy(len(slice_mdp.states), 3)
        for j in range(SIMULATIONS):
            random_simulation = SliceSimulator(arrivals, departures, c_lost=2, simulation_time=1000, verbose=False)
            random_agent = Agent(slice_mdp.states, random_policy, random_simulation)
            random_stats_tmp.append(random_agent.control_environment())
        random_stats.append({'costs_per_timeslot': get_mean_costs(random_stats_tmp)['mean'],
                             'processed_jobs_per_timeslot': get_mean_processed_jobs(random_stats_tmp)['mean'],
                             'lost_jobs_per_timeslot': get_mean_lost_jobs(random_stats_tmp)['mean']})

    print(f"Simulation done in {(time.time() - time_start) / 60} minutes")

    # plotting!
    easy_plot("mdp-toy", mdp_stats, True)
    easy_plot("random-toy", random_stats, True)


