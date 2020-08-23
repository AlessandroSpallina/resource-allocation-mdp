from agent import Agent
from slice_mdp import SliceMDP
from slice_simulator import SliceSimulator

if __name__ == '__main__':
    sim = SliceSimulator([0.5, 0.5], [0.6, 0.4], 2, c_lost=10, simulation_time=1000, verbose=False)
    mdp = SliceMDP([0.5, 0.5], [0.6, 0.4], 2, 1, c_lost=10)
    agent = Agent(mdp.states, (0, 0, 1, 0, 0, 0), sim)
    # agent = Agent(mdp.states, (1, 1, 1, 0, 0, 0), sim)
    print(agent.control_environment()['wait_time_per_job'])
    # print(utils.read_config(True))
