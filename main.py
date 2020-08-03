from slice_mdp import SliceMDP
from agent import Agent
from slice_simulator import SliceSimulator
import plotter
import numpy as np

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


if __name__ == '__main__':
    arrivals = [0.5, 0.5]
    departures = [0.6, 0.4]
    stats = []

    # mdp agent
    slice_mdp = SliceMDP(arrivals, departures, 2, 1, alpha=0.5, c_lost=2)
    plotter.plot_markov_chain(slice_mdp.states, slice_mdp.transition_matrix, slice_mdp.reward_matrix,
                              projectname="toy", view=False)

    for i in range(SIMULATIONS):
        slice_simulator = SliceSimulator(arrivals, departures, c_lost=2, simulation_time=10000)
        mdp_agent = Agent(slice_mdp.states, slice_mdp.run_value_iteration(0.8), slice_simulator)
        stats.append(mdp_agent.control_environment())

    #  print(stats)

    mean_costs = get_mean_costs(stats)
    mean_processed_jobs = get_mean_processed_jobs(stats)
    mean_lost_jobs = get_mean_lost_jobs(stats)

    # plotting of one simulation
    plotter.plot_cumulative({"costs": stats[0]['costs_per_timeslot'],
                             "processed": stats[0]['processed_jobs_per_timeslot'],
                             "lost": stats[0]['lost_jobs_per_timeslot']},
                            title="One Sim Cumulative", projectname="toy", view=False)
    plotter.plot({"costs": stats[0]['costs_per_timeslot'],
                  "processed": stats[0]['processed_jobs_per_timeslot'],
                  "lost": stats[0]['lost_jobs_per_timeslot']},
                 title="One Sim per Timeslot", projectname="toy", view=False)

    # plotting of the mean of N simulations
    plotter.plot_cumulative({"costs": mean_costs['mean'],
                             "processed": mean_processed_jobs['mean'],
                             "lost": mean_lost_jobs['mean']},
                            title="Mean Cumulative", projectname="toy", view=False)
    plotter.plot({"costs per ts": mean_costs['mean'],
                  "processed per ts": mean_processed_jobs['mean'],
                  "lost per ts": mean_lost_jobs['mean']},
                 title="Mean per Timeslot", projectname="toy", view=False)

