from refactoring.config import PolicyConfig, EnvironmentConfig
from refactoring.policy import MultiSliceMdpPolicy
from refactoring.environment import MultiSliceSimulator
from refactoring.agent import NetworkOperator


def main():
    policy_conf = PolicyConfig()
    policy = MultiSliceMdpPolicy(policy_conf)
    policy.calculate_policy()

    environment_conf = EnvironmentConfig()
    environment = MultiSliceSimulator(environment_conf)

    agent = NetworkOperator(policy, environment, policy_conf.timeslots)
    agent.start_automatic_control()

    print(agent.history)

if __name__ == "__main__":
    main()
