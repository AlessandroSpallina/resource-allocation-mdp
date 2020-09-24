import matplotlib.pyplot as plt

from src.slicing_core import utils, slice_simulator as q

if __name__ == '__main__':
    slice = q.QueueSimulator(29, 30, 30, 5)

    # plotting incoming jobs
    plt.bar(range(slice.simulation_time), slice.incoming_jobs)
    plt.title("Incoming Jobs")
    plt.xlabel("Timeslot")
    plt.ylabel("Job Count")
    plt.show()

    # slice.allocate_server()

    for i in range(slice.simulation_time):
        total_processed_jobs, jobs_in_queue, server_count = slice.simulate_timeslot(verbose=True)
        utils.print_blue("[T " + str(i) + "]Total Processed Jobs: " + total_processed_jobs
                         + "Jobs in Queue: " + jobs_in_queue
                         + ", Server Count: " + server_count)
