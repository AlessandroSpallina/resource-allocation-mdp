# TODO:
# * consider using cupy for gpu acceleration
# * consider using generators, see https://wiki.python.org/moin/Generators

import numpy as np
import matplotlib.pyplot as plt
import queue


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


class QueueSimulator:
    def __init__(self, arrival_rate, departure_rate, simulation_time, max_server_number_cap=100):
        self._arrival_rate = arrival_rate
        self._departure_rate = departure_rate
        self._simulation_time = simulation_time
        self._max_server_number_cap = max_server_number_cap
        # server_num: dynamic value
        self._server_num = 1
        # rho: lambda / miu * servers
        self._rho = self._update_rho()
        # incoming jobs: index i represents timeslot i, value[i] represents the number of jobs
        self._incoming_jobs = np.random.poisson(arrival_rate, simulation_time)
        # service time: index i represents service time for i-esimo job
        self._service_time = np.random.exponential(1 / departure_rate, self._incoming_jobs.sum())
        self._terminated_jobs = 0
        self._simulation_timeslot = 0 # contains the current timeslot
        self._queue = queue.Queue()
        # server_time: lista contenente N elementi, ogni elemento indica l'utilizzo (in frazione di timeslot) del server
        # [0, 1, 0.5] indica che il primo server è occupato per tutto il timeslot, il secondo è idle,
        # il terzo è occupato per metà timeslot
        self._server_time = [1.0]

    def _update_rho(self):
        self._rho = self._arrival_rate / (self._departure_rate * self._server_num)
        return self._rho

    @property
    def incoming_jobs(self):
        return self._incoming_jobs

    """
    Return the period of simulation (i.e. 1000 timeslot)
    """
    @property
    def simulation_time(self):
        return self._simulation_time

    """
    func allocate_server returns rho if success, otherwise raise ServerMaxCapError 
    """
    def allocate_server(self, count=1):
        if self._server_num + count > self._max_server_number_cap:
            raise ServerMaxCapError('Max Cap Limit Reached',
                                    'Unable to allocate {self._server_num + count} servers; '
                                    'max cap is {self._max_server_number_cap}')
        self._server_num += count
        self._server_time.append(1.0)
        return self._update_rho()

    """
    func deallocate_server returns rho if success, otherwise raise ServerMinCapError 
    """
    def deallocate_server(self, count=1):
        if self._server_num - count < 1:
            raise ServerMinCapError('Min Cap Limit Reached',
                                    'Unable to deallocate to {self._server_num + count} servers; min cap is 1')
        self._server_num -= count
        # fixme: dovresti deallocare una vm idle, altrimenti perdi il job in esecuzione!!
        #        e quindi gestire il problema (ritardare la deallocazione) se non trovi vm idle
        self._server_time.pop()
        return self._update_rho()

    # this can be (probably) huge optimized
    def simulate_timeslot(self, verbose=False):
        # estrarre 1 elemento da self._incoming_jobs (numero di job arrivati in un timeslot)
        # possono lavorare N server in parallelo
        if self._incoming_jobs.size > 0:
            job_num = self._incoming_jobs[0]
            self._incoming_jobs = np.delete(self._incoming_jobs, 0)  # cancella il primo elemento
            # aggiungi alla coda gli n job ricevuti in questo time slot
            j = Job(self._simulation_timeslot)
            for i in range(job_num):
                self._queue.put(j)
            if verbose:
                print(f"At timeslot {self._simulation_timeslot} the queue has {self._queue.qsize()} pending job")

            # se ci sono server liberi, che prendano un task in carico!
            for i in range(self._server_num):
                if verbose:
                    print(f"Server[{i}] in execution (current timeslot {self._simulation_timeslot})")
                if self._server_time[i] > 0:  # server con tempo utile :)
                    # NB. in un timeslot posso potenzialmente processare più job in un timeslot
                    while self._server_time[i] > 0:
                        if verbose:
                            print(f"Server[{i}] has {self._server_time[i] * 100}% of the timeslot free")

                        # estraggo un job dalla coda
                        # ottenere delle statistiche empiriche(?), es. tempo in coda di questo task
                        try:
                            tmp = self._queue.get(False)
                        except queue.Empty:
                            pass

                        # estraggo un service time, il server ci impiegerà quel tempo per processare il job
                        # quindi il tempo utile in _server_time[i] diminuisce
                        self._server_time[i] -= self._service_time[0]
                        self._service_time = np.delete(self._service_time, 0)
                        self._terminated_jobs += 1

            self._simulation_timeslot += 1
            # il tempo è passato, dò un timeslot a tutti i server
            for i in range(self._server_num):
                self._server_time[i] += 1
                if verbose:
                    print(f"Server[{i}] can work the {self._server_time[i]}% of the next timeslot")

        return {"total_processed_jobs": self._terminated_jobs,
                "jobs_in_queue": self._queue.qsize(),
                "server_count": self._server_num}
