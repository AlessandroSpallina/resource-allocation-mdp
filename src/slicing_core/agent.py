class NetworkOperator:
    def __init__(self, policy, environment, control_timeslot_duration):
        self._policy = policy
        self._environment = environment
        self._control_timeslot_duration = control_timeslot_duration

        self._current_timeslot = 0
        self._current_state = self._environment.current_state

        # statistics
        self._history = []

    @property
    def history(self):
        return self._history

    def start_automatic_control(self):
        for i in range(self._control_timeslot_duration):
            self._current_timeslot = i
            self._current_state = self._environment.current_state
            action_to_do = self._policy.get_action_from_policy(self._current_state, self._current_timeslot)
            self._history.append(self._environment.next_timeslot(action_to_do))


