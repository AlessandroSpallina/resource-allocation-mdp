# TODO:
# * consider using cupy for gpu acceleration
# * consider using generators, see https://wiki.python.org/moin/Generators

import numpy as np
import queue
from slice_mdp import State
import random
import numpy as np


class Error(Exception):
    pass


class ServerMaxCapError(Error):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class ServerMinCapError(Error):
    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class Job:
    def __init__(self, arrival_timeslot):
        self._arrival_timeslot = arrival_timeslot

    @property
    def arrival_timeslot(self):
        return self._arrival_timeslot


class SliceSimulator:
    def __init__(self, arrivals_histogram, departures_histogram, queue_size=2, simulation_time=100, max_server_num=1):
        self._arrivals_histogram = arrivals_histogram
        self._departures_histogram = departures_histogram
        self._simulation_time = simulation_time
        self._max_server_num = max_server_num
        self._queue_size = queue_size

        self._current_state = State(0, 0)
        self._h_p = self._generate_h_p()
        self._current_timeslot = 0  # contains the current timeslot

        # incoming jobs: index i represents timeslot i, value[i] represents the number of jobs
        self._incoming_jobs = self._generate_incoming_jobs()

        self._queue = queue.Queue(queue_size)

    @property
    def incoming_jobs(self):
        return self._incoming_jobs

    def _generate_h_p(self):
        h_p = []
        for i in range(self._max_server_num + 1):
            if i == 0:
                h_p.append([])
            elif i == 1:
                h_p.append(self._departures_histogram)
            else:
                h_p.append(np.convolve(self._departures_histogram, h_p))
        return h_p

    # returns an array, each element represent the num of jobs arrived in the timeslot
    def _generate_incoming_jobs(self):
        incoming_jobs = []
        for i in range(self._simulation_time):
            prob = random.random()  # genera valore random tra [0., 1.[
            for j in range(len(self._arrivals_histogram)):
                if prob <= self._arrivals_histogram[j]:
                    # in questo ts arrivano j jobs
                    incoming_jobs.append(j)
                else:
                    prob -= self._arrivals_histogram[j]
        return incoming_jobs

    # returns the number of jobs processed in one timeslot
    def _calculate_processed_jobs(self):
        prob = random.random()
        for j in range(len(self._h_p[self._current_state.n])):
            if prob <= self._h_p[self._current_state.n][j]:
                return j
            else:
                prob -= self._h_p[self._current_state.n][j]

    def _allocate_server(self, count=1):
        if self._current_state.n + count > self._max_server_num:
            raise ServerMaxCapError('Max Cap Limit Reached',
                                    'Unable to allocate {self._server_num + count} servers; '
                                    'max cap is {self._max_server_number_cap}')
        self._current_state.n += count

    def _deallocate_server(self, count=1):
        if self._current_state.n - count < 0:
            raise ServerMinCapError('Min Cap Limit Reached',
                                    'Unable to deallocate to {self._server_num + count} servers; min cap is 1')
        self._current_state.n -= count

    # returns the current state
    def simulate_timeslot(self, action_id, verbose=False):
        if verbose:
            print(f"[TS{self._current_timeslot}] Current state: {self._current_state}")

        if len(self._incoming_jobs) > 0:
            arrived_jobs = self._incoming_jobs.pop()

            if verbose:
                print(f"[TS{self._current_timeslot}] Arrived {arrived_jobs} jobs")

            j = Job(self._current_timeslot)
            for i in range(arrived_jobs):
                try:
                    self._queue.put(j, False)
                    self._current_state.k += 1
                except queue.Full:
                    if verbose:
                        print(f"[TS{self._current_timeslot}] Lost packet here")

            if verbose:
                print(f"[TS{self._current_timeslot}] The queue has {self._queue.qsize()} pending job")

            if self._current_state.n > 0 and self._queue.qsize() > 0:  # allora c'è processamento
                processed_jobs = self._calculate_processed_jobs()

                if verbose:
                    print(f"[TS{self._current_timeslot}] Processed {processed_jobs} jobs")

                for i in range(processed_jobs):
                    try:
                        job = self._queue.get(False)
                        self._current_state.k -= 1
                        # todo: collezionare statistiche!
                    except queue.Empty:
                        pass

            if action_id == 1:
                self._allocate_server()
            elif action_id == 2:
                self._deallocate_server()

            self._current_timeslot += 1

        return self._current_state

