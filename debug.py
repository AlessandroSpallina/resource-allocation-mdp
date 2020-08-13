from slice_simulator import SliceSimulator
import main

if __name__ == '__main__':
    slice_sim = SliceSimulator([0.5,0.5], [0.6,0.4], c_lost=2, simulation_time=50, verbose=True)

    print(f"Incoming jobs {slice_sim._incoming_jobs}")

    print(slice_sim.simulate_timeslot(1))
    for i in range(49):
        print(slice_sim.simulate_timeslot(0))

    stats = slice_sim.get_statistics()

    main.easy_plot('debug', [stats], True)
