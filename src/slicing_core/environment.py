import abc
import queue
import numpy as np
import random
from copy import copy

from src.slicing_core.state import SingleSliceState


class Environment(metaclass=abc.ABCMeta):
    @property
    @abc.abstractmethod
    def current_state(self):
        pass

    def __init__(self, environment_config):
        self._config = environment_config

    @abc.abstractmethod
    def next_timeslot(self, slices_allocation):
        pass


class Job:
    def __init__(self, arrival_timeslot):
        self._arrival_timeslot = arrival_timeslot

    @property
    def arrival_timeslot(self):
        return self._arrival_timeslot


class SingleSliceSimulator(Environment):
    def __init__(self, environment_config):
        super().__init__(environment_config)

        self._current_state = SingleSliceState(0, 0)
        self._current_timeslot = 0
        self._queue = queue.Queue(self._config.queue_size)
        self._servers_internal_queue = queue.Queue(self._config.server_max_cap)
        self._state_sequence = []

        self._generate_incoming_jobs()
        self._generate_h_d()

    @property
    def current_state(self):
        return self._current_state

    """
    Control the next timeslot imposing servers allocation to all slices.
    Returns a snapshot of the ended timeslot
    """
    def next_timeslot(self, slice_allocation):
        if self._config.immediate_action:
            self._allocate_server(slice_allocation)

        # statistics
        initial_state = copy(self._current_state)
        self._state_sequence.append(initial_state)

        if self._config.arrival_processing_phase:
            arrived_losses = self._simulate_arrival_phase_within_timeslot()
            processing = self._simulate_processing_phase_within_timeslot()
        else:
            processing = self._simulate_processing_phase_within_timeslot()
            arrived_losses = self._simulate_arrival_phase_within_timeslot()

        if not self._config.immediate_action:
            self._allocate_server(slice_allocation)

        self._current_timeslot += 1

        return {
            "timeslot": self._current_timeslot,
            "state": copy(self._current_state),
            "active_servers": self._current_state.n,
            "jobs_in_queue": self._current_state.k,
            "jobs_in_system": self._current_state.k + self._servers_internal_queue.qsize(),
            "incoming_jobs": arrived_losses['arrivals'],
            "lost_jobs": arrived_losses['losses'],
            "processed_jobs": processing['processed_jobs'],
            "wait_time_in_the_queue": processing['wait_time_in_the_queue'],
            "wait_time_in_the_system": processing['wait_time_in_the_system']
        }

    def _generate_h_d(self):
        self._h_d = []
        for i in range(self._config.server_max_cap + 1):
            if i == 0:
                self._h_d.append([])
            elif i == 1:
                self._h_d.append(self._config.server_capacity_histogram)
            else:
                self._h_d.append(
                    np.convolve(self._config.server_capacity_histogram, self._h_d[i - 1]).tolist()
                )

    """ Returns an array, each element represent the num of jobs arrived in the timeslot """
    def _generate_incoming_jobs(self):
        self._incoming_jobs = [0]
        for _ in range(self._config.timeslots):
            prob = random.random()  # generate a random value [0., 1.[
            for j in range(len(self._config.arrivals_histogram)):
                if prob <= self._config.arrivals_histogram[j]:
                    # in this ts arrive j jobs
                    self._incoming_jobs.append(j)
                    break
                else:
                    prob -= self._config.arrivals_histogram[j]
        # in simulation we use .pop() that give the last inserted item
        # with a .reverse() we can impose that the first arrive is 0 jobs
        self._incoming_jobs.reverse()

    """ This function allocate server_num servers, this is an absolute number (not a delta) """
    def _allocate_server(self, server_num):
        if 0 <= server_num <= self._config.server_max_cap:
            self._current_state.n = server_num

    """ 
    Moves jobs from the queue to the internal servers cache.
    Returns the number of jobs moved with success
    """
    def _refill_server_internal_queue(self):
        refill_counter = 0
        wait_time_in_the_queue = []

        for _ in range(self._current_state.n - self._servers_internal_queue.qsize()):
            # extract from the queue and put the job in server queue (the cache)
            try:
                job = self._queue.get(False)
                self._current_state.k -= 1
                wait_time_in_the_queue.append(self._current_timeslot - job.arrival_timeslot)
                self._servers_internal_queue.put(job, False)
                refill_counter += 1
            except queue.Empty:
                pass

        return {"refilled": refill_counter, "wait_time_in_the_queue": wait_time_in_the_queue}

    """ Returns the number of jobs processed in one timeslot """
    def _calculate_processed_jobs(self):
        prob = random.random()
        for j in range(len(self._h_d[self._current_state.n])):
            if prob <= self._h_d[self._current_state.n][j]:
                return j
            else:
                prob -= self._h_d[self._current_state.n][j]

    """ Returns the number of lost jobs in the timeslot """
    def _simulate_arrival_phase_within_timeslot(self):
        lost_counter = 0
        arrived_jobs = 0
        if len(self._incoming_jobs) > 0:
            arrived_jobs = self._incoming_jobs.pop()

            j = Job(self._current_timeslot)

            for _ in range(arrived_jobs):
                try:
                    self._queue.put(j, False)
                    self._current_state.k += 1
                except queue.Full:
                    lost_counter += 1
        return {'arrivals': arrived_jobs, 'losses': lost_counter}

    """ Returns the number of processed jobs in the timeslot """
    def _simulate_processing_phase_within_timeslot(self):
        processed_jobs = 0
        wait_time_in_the_queue = []
        wait_time_in_the_system = []

        if self._current_state.n > 0:  # if here, i can process jobs
            # estrazione dei job dalla coda e inserimento in cache server (sono in run)
            # poi, processo i job che sono in cache
            # se all'interno dello stesso timeslot posso processare più dei job in cache, refillo la cache
            wait_time_in_the_queue = self._refill_server_internal_queue()['wait_time_in_the_queue']
            processed_jobs = self._calculate_processed_jobs()
            for _ in range(processed_jobs):
                try:
                    job = self._servers_internal_queue.get(False)
                    wait_time_in_the_system.append(self._current_timeslot - job.arrival_timeslot)
                except queue.Empty:
                    tmp = self._refill_server_internal_queue()
                    wait_time_in_the_queue += tmp['wait_time_in_the_queue']
                    if tmp['refilled'] == 0:
                        # se entro qui posso processare più job di quanti ne ho in coda ->
                        # la stat deve essere relativa al reale processato!!!!
                        processed_jobs -= 1
        return {
            'processed_jobs': processed_jobs,
            'wait_time_in_the_queue': wait_time_in_the_queue,
            'wait_time_in_the_system': wait_time_in_the_system
        }


class MultiSliceSimulator(Environment):
    def __init__(self, environment_config):
        super().__init__(environment_config)
        self._current_state = [SingleSliceState(0, 0) for _ in range(self._config.slice_count)]
        self._init_simulations()

    @property
    def current_state(self):
        return self._current_state

    def next_timeslot(self, slices_allocation):
        tmp = []
        for i in range(self._config.slice_count):
            tmp.append(self._simulations[i].next_timeslot(slices_allocation[i]))
        self._current_state = [s['state'] for s in tmp]
        return {
            "timeslot": tmp[0]['timeslot'],
            "state":  copy(self._current_state),
            "active_servers": [s['active_servers'] for s in tmp],
            "jobs_in_queue": [s['jobs_in_queue'] for s in tmp],
            "jobs_in_system": [s['jobs_in_system'] for s in tmp],
            "incoming_jobs": [s['incoming_jobs'] for s in tmp],
            "lost_jobs": [s['lost_jobs'] for s in tmp],
            "processed_jobs": [s['processed_jobs'] for s in tmp],
            "wait_time_in_the_queue": [s['wait_time_in_the_queue'] for s in tmp],
            "wait_time_in_the_system": [s['wait_time_in_the_system'] for s in tmp]
        }

    # this should be renamed to _init_slices because it creates N single-slice-simulator
    def _init_simulations(self):
        self._simulations = []
        for i in range(self._config.slice_count):
            self._simulations.append(SingleSliceSimulator(self._config.slice(i)))
