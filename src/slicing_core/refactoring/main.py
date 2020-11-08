from refactoring.config import PolicyConfig, EnvironmentConfig
from refactoring.policy import MultiSliceMdpPolicy
from refactoring.environment import MultiSliceSimulator
from refactoring.agent import NetworkOperator

import plotter
import time


def main():
    policy_conf = PolicyConfig()

    start_time = time.time()

    policy = MultiSliceMdpPolicy(policy_conf)

    print(f"Initialization done in {time.time() - start_time} seconds")
    start_time = time.time()

    policy.calculate_policy()

    print(f"Policy calculation done in {time.time() - start_time} seconds")

    environment_conf = EnvironmentConfig()
    environment = MultiSliceSimulator(environment_conf)

    start_time = time.time()

    agent = NetworkOperator(policy, environment, policy_conf.timeslots)
    agent.start_automatic_control()

    print(f"Simulation done in {time.time() - start_time} seconds")

    history = agent.history

    # plotting system servers vs job in the queue vs lost jobs vs processed jobs
    # allocated_servers = []
    # for timeslot in history:
    #     tmp = []
    #     for substate in timeslot['state']:
    #         tmp.append(substate.n)
    #     allocated_servers.append(sum(tmp))
    #
    # job_in_queue = []
    # for timeslot in history:
    #     tmp = []
    #     for substate in timeslot['state']:
    #         tmp.append(substate.k)
    #     job_in_queue.append(sum(tmp))
    #
    # lost_jobs = []
    # for timeslot in history:
    #     lost_jobs.append(timeslot['lost_jobs'])
    #
    # processed_jobs = []
    # for timeslot in history:
    #     processed_jobs.append(timeslot['processed_jobs'])
    #
    # plotter.plot(
    #     ydata={
    #         "system allocated servers": allocated_servers,
    #         "job in the queue": job_in_queue,
    #         # "lost jobs": lost_jobs,
    #         # "processed jobs": processed_jobs
    #     },
    #     xdata=list(range(len(history))),
    #     xlabel="timeslot", title="", projectname="hello-refactoring", view=False
    # )

    policy_stuct = policy.policy
    states = policy.states

    # plotter.table([f'{states[i][0].k},{states[i][1].k} jobs' for i in range(len(policy_stuct))],
    #               [f'{states[i][0].n},{states[i][1].n} servers' for i in range(len(policy_stuct))],
    #               [actions[policy_stuct[i]] for i in range(len(policy_stuct))], title="policy", projectname="hello-refactoring")

    tmp_1 = [f'{states[i][0].k},{states[i][1].k} jobs' for i in range(len(policy_stuct))]
    tmp_2 = [f'{states[i][0].n},{states[i][1].n} servers' for i in range(len(policy_stuct))]
    tmp_3 = [policy_stuct[i] for i in range(len(policy_stuct))]

    for i in range(len(policy_stuct)):
        print(tmp_1[i], tmp_2[i], tmp_3[i])


if __name__ == "__main__":
    main()
