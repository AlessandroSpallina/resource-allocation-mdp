from refactoring.config import PolicyConfig
from refactoring.policy import MultiSliceMdpPolicy

def main():
    policy_conf = PolicyConfig()
    mdp = MultiSliceMdpPolicy(policy_conf)
    mdp.calculate_policy()
    print("ok")




    # concrete_strategy_a = ConcreteStrategyA()
    # context = Context(concrete_strategy_a)
    # context.context_interface()


if __name__ == "__main__":
    main()
