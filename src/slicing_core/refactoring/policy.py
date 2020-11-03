import abc


class Policy(metaclass=abc.ABCMeta):
    """
    Declare an interface common to all supported algorithms. Context
    uses this interface to call the algorithm defined by a
    ConcreteStrategy.
    """

    @abc.abstractmethod
    def __init__(self, policy_config):
        pass

    @abc.abstractmethod
    def calculate_policy(self):
        pass

    @abc.abstractmethod
    def get_action_from_policy(self, current_state):
        pass


class MdpPolicy(Policy):
    def __init__(self, policy_config):
        self._config = policy_config

    def calculate_policy(self):
        pass

    def get_action_from_policy(self, current_state):
        pass


class ConservativePolicy(Policy):
    def __init__(self, policy_config):
        self._config = policy_config

    def calculate_policy(self):
        pass

    def get_action_from_policy(self, current_state):
        pass

