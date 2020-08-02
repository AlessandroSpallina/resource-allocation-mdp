from slice_mdp import SliceMDP
from slice_simulator import SliceSimulator
import utils

if __name__ == '__main__':
    arrivals = [0.5, 0.5]
    departures = [0.6, 0.4]

    slice_mdp = SliceMDP(arrivals, departures, 2, 1, alpha=0.5, c_lost=2)
    states = slice_mdp.states

    slice_mdp.plot("toy", True)

    policy = slice_mdp.run_value_iteration(0.8)
    print(policy)

    slice_simulator = SliceSimulator(arrivals, departures)

    action = 0
    for i in range(100):
        current_state = slice_simulator.simulate_timeslot(action, True)
        print(f"RETURNED {current_state}")
        action = policy[states.index(current_state)]

    # statistics

    print("STATISTICS")
    print(slice_simulator.get_statistics())


