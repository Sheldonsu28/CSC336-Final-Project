import numpy as np
from matplotlib import pyplot as plt
import time

# This code was written few month after the course has ended




G = 6.67e-11

#AU = 1.496e11
fl = np.float64

def simulation(simulation_length, dt):
    """
    Generate The plot for simulation using velocity verlet
    :param simulation_length: Real time in seconds
    :param dt: time step
    :return: None
    """

    print("Initializing...")
    simulation_length = int(simulation_length/dt)
    # Sun mercury venus earth mars jupiter saturn uranus neptune
    entity_position_x = np.zeros(simulation_length * 9 + 9, dtype=np.float64).reshape(9, simulation_length + 1)
    entity_position_y = np.zeros(simulation_length * 9 + 9, dtype=np.float64).reshape(9, simulation_length + 1)

    entity_velocity_x = np.zeros(simulation_length * 9 + 9, dtype=np.float64).reshape(9, simulation_length + 1)
    entity_velocity_y = np.zeros(simulation_length * 9 + 9, dtype=np.float64).reshape(9, simulation_length + 1)

    entity_acc_x = np.zeros(simulation_length * 9 + 9, dtype=np.float64).reshape(9, simulation_length + 1)
    entity_acc_y = np.zeros(simulation_length * 9 + 9, dtype=np.float64).reshape(9, simulation_length + 1)

    entity_mass = np.array([1988500e24, 0.33011e24, 4.8675e24, 5.9723e24, 0.64171e24, 1898.19e24, 568.34e24, 86.813e24,
                            102.413e24], dtype=np.float64)

    entity_position_x[:, 0] = np.array([0, 69.82e9, 108.94e9, 152.10e9, 249.23e9, 816.62e9, 1514.50e9,
                                        3003.62e9, 4545.67e9], dtype=np.float64)

    entity_velocity_y[:, 0] = np.array([0, 38.86e3, 34.79e3, 29.29e3, 21.97e3, 12.44e3, 9.09e3,
                                        6.49e3, 5.37e3], dtype=np.float64)

    print("Simulating...")
    a = time.time()

    x_diff = np.empty((9, ), dtype=np.float64)
    y_diff = np.empty((9, ), dtype=np.float64)

    x_diff_1 = np.empty((9,), dtype=np.float64)
    y_diff_1 = np.empty((9,), dtype=np.float64)

    for i in range(0, simulation_length):

        for p in range(1, 9):
            x_diff.fill(entity_position_x[p, i])
            x_diff -= entity_position_x[:, i]
            x_diff[p] = 2

            y_diff.fill(entity_position_y[p, i])
            y_diff -= entity_position_y[:, i]
            y_diff[p] = 2
            r_cube = np.power(np.sqrt(np.square(x_diff) + np.square(y_diff)), 3)

            acc_list_x = -G * entity_mass * x_diff / r_cube
            acc_list_y = -G * entity_mass * y_diff / r_cube
            acc_list_x[p] = 0
            acc_list_y[p] = 0

            entity_acc_x[p, i] = np.sum(acc_list_x)
            entity_acc_y[p, i] = np.sum(acc_list_y)

            entity_position_x[p, i + 1] = entity_position_x[p, i] + (entity_velocity_x[p, i] +
                                                                     0.5 * entity_acc_x[p, i] * dt)*dt
            entity_position_y[p, i + 1] = entity_position_y[p, i] + (entity_velocity_y[p, i] +
                                                                     0.5 * entity_acc_y[p, i] * dt)*dt


        for pl in range(1, 9):
            x_diff_1.fill(entity_position_x[pl, i + 1])
            x_diff_1 -= entity_position_x[:, i + 1]
            x_diff_1[pl] = 2
            y_diff_1.fill(entity_position_y[pl, i + 1])
            y_diff_1 -= entity_position_y[:, i + 1]
            y_diff_1[pl] = 2

            r_cube_1 = np.power(np.sqrt(np.square(x_diff_1) + np.square(y_diff_1)), 3)

            acc_x = -G * entity_mass * x_diff_1 / r_cube_1
            acc_y = -G * entity_mass * y_diff_1 / r_cube_1

            acc_x[pl] = 0
            acc_y[pl] = 0

            entity_acc_x[pl, i + 1] = np.sum(acc_x)
            entity_acc_y[pl, i + 1] = np.sum(acc_y)

            entity_velocity_x[pl, i + 1] = entity_velocity_x[pl, i] + \
                                           0.5 * (entity_acc_x[pl, i + 1] + entity_acc_x[pl, i]) * dt

            entity_velocity_y[pl, i + 1] = entity_velocity_y[pl, i] + \
                                           0.5 * (entity_acc_y[pl, i + 1] + entity_acc_y[pl, i]) * dt


    c = time.time()
    print("Simulation took:", c - a, "s to finish")
    result = {'mercury': [entity_position_x[1, :], entity_position_y[1, :]],
              'venus': [entity_position_x[2, :], entity_position_y[2, :]],
              'earth': [entity_position_x[3, :], entity_position_y[3, :]],
              'mars': [entity_position_x[4, :], entity_position_y[4, :]],
              'jupiter': [entity_position_x[5, :], entity_position_y[5, :]],
              'saturn': [entity_position_x[6, :], entity_position_y[6, :]],
              'uranus': [entity_position_x[7, :], entity_position_y[7, :]],
              'neptune': [entity_position_x[8, :], entity_position_y[8, :]],
              'sun': [entity_position_x[0, :], entity_position_y[0, :]]}
    for p in result.keys():
        if p == 'sun':
            plt.plot(result[p][0], result[p][1], 'ro', label=p)
        else:
            plt.plot(result[p][0], result[p][1], label=p)
    plt.legend()
    plt.title("Earth trajectory after acceleration a = 4e-5 for 17 years")
    plt.xlabel("distance, unit: m")
    plt.ylabel("distance, unit: m")
    plt.savefig("4e-5 years.png")
    plt.show()


if __name__ == "__main__":
    simulation(3.154e+7, 60)
