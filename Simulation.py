import numpy as np
from matplotlib import pyplot as plt
import time

G = 6.67e-11


def simulation_verlet(simulation_time, dt):
    """
    Simulate planetory motion using Velocity verlet methods
    :param simulation_time: Total simulation time (in seconds)
    :param dt: time step (in seconds)
    """

    print("Initializing...")
    simulation_length = int(simulation_time / dt)
    # Initialize the data array to hold the motion data using the following order:
    # Sun mercury venus earth mars jupiter satern uranus neptune

    entity_position_x = np.zeros(simulation_length * 9 + 9, dtype=np.longdouble).reshape(9, simulation_length + 1)
    entity_position_y = np.zeros(simulation_length * 9 + 9, dtype=np.longdouble).reshape(9, simulation_length + 1)

    entity_velocity_x = np.zeros(simulation_length * 9 + 9, dtype=np.longdouble).reshape(9, simulation_length + 1)
    entity_velocity_y = np.zeros(simulation_length * 9 + 9, dtype=np.longdouble).reshape(9, simulation_length + 1)

    entity_acc_x = np.zeros(simulation_length * 9 + 9, dtype=np.longdouble).reshape(9, simulation_length + 1)
    entity_acc_y = np.zeros(simulation_length * 9 + 9, dtype=np.longdouble).reshape(9, simulation_length + 1)

    entity_mass = np.array([1988500e24, 0.33011e24, 4.8675e24, 5.9723e24, 0.64171e24, 1898.19e24, 568.34e24, 86.813e24,
                            102.413e24], dtype=np.longdouble)

    # Set up the initial position and velocity data
    entity_position_x[:, 0] = np.array([0, 69.82e9, 108.94e9, 152.10e9, 249.23e9, 816.62e9, 1514.50e9,
                                        3003.62e9, 4545.67e9], dtype=np.longdouble)

    entity_velocity_y[:, 0] = np.array([0, 38.86e3, 34.79e3, 29.29e3, 21.97e3, 12.44e3, 9.09e3,
                                        6.49e3, 5.37e3], dtype=np.longdouble)

    print("Simulating...")
    print("There are {} simulation steps in total".format(simulation_length))
    a = time.time()
    for i in range(0, simulation_length):

        # Did not calculate the motion of the Sun since Sun will be in the center of the graph
        # Perform verlet integration step for all planets
        for p in range(1, 9):
            entity_position_x[p, i + 1] = entity_position_x[p, i] + (entity_velocity_x[p, i] +
                                                                     0.5 * entity_acc_x[p, i] * dt) * dt
            entity_position_y[p, i + 1] = entity_position_y[p, i] + (entity_velocity_y[p, i] +
                                                                     0.5 * entity_acc_y[p, i] * dt) * dt

        for pl in range(1, 9):
            entity_acc_x[pl, i + 1], entity_acc_y[pl, i + 1] = calc_acc(entity_position_x[pl, i + 1],
                                                                        entity_position_x[:, i + 1],
                                                                        entity_position_y[pl, i + 1],
                                                                        entity_position_y[:, i + 1], pl, entity_mass)

            entity_velocity_x[pl, i + 1] = entity_velocity_x[pl, i] + \
                                           0.5 * (entity_acc_x[pl, i + 1] + entity_acc_x[pl, i]) * dt

            entity_velocity_y[pl, i + 1] = entity_velocity_y[pl, i] + \
                                           0.5 * (entity_acc_y[pl, i + 1] + entity_acc_y[pl, i]) * dt

    c = time.time()
    print("Verlet simulation took:", c - a, "s to finish")

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

    diff = calculate_energy_diff(entity_position_x, entity_position_y, entity_velocity_x, entity_velocity_y,
                                 entity_mass)
    print("The energy difference for Verlet method is:{}%".format(diff))
    plt.legend()
    plt.title("Verlet Planet Trajectory simulation for {} years".format(simulation_time / 3.154e+7))
    plt.xlabel("distance, unit: m")
    plt.ylabel("distance, unit: m")
    plt.savefig("Verlet {}.png".format(simulation_time / 3.154e+7))


def calculate_energy_diff(xp, yp, xv, yv, m):
    initial_KE = np.sum(0.5 * m[1:] * ((xv[1:, 0] ** 2) + (yv[1:, 0] ** 2)))
    Final_KE = np.sum(0.5 * m[1:] * ((xv[1:, -1] ** 2) + (yv[1:, -1] ** 2)))

    r_initial = np.sqrt((xp[1:, 0] ** 2) + (yp[1:, 0] ** 2))
    r_final = np.sqrt((xp[1:, -1] ** 2) + (yp[1:, -1] ** 2))

    inital_U = np.sum(G * m[0] * (m[1:] / r_initial))
    Final_U = np.sum(G * m[0] * (m[1:] / r_final))

    return abs(((initial_KE + inital_U) - (Final_KE + Final_U)))*100/(initial_KE + inital_U)


def calc_acc(px, other_px, py, other_py, p, mass):
    x_diff = px - other_px
    # Assign the current planet with some constant to avoid division by zero error
    x_diff[p] = 16
    y_diff = py - other_py
    y_diff[p] = 16
    r_cube = np.power(np.sqrt(np.square(x_diff) + np.square(y_diff)), 3)

    acc_list_x = -G * mass * x_diff / r_cube
    acc_list_y = -G * mass * y_diff / r_cube
    acc_list_x[p] = 0
    acc_list_y[p] = 0
    return np.sum(acc_list_x), np.sum(acc_list_y)
    # return 0, 0


def simulation_Runge_kutta_4rd(simulation_time, dt):
    """
    Simulate planetory motion using Velocity verlet methods
    :param simulation_time: Total simulation time (in seconds)
    :param dt: time step (in seconds)
    """

    print("Initializing...")
    simulation_length = int(simulation_time / dt)
    # Initialize the data array to hold the motion data using the following order:
    # Sun mercury venus earth mars jupiter satern uranus neptune

    entity_position_x = np.zeros(simulation_length * 9 + 9, dtype=np.longdouble).reshape(9, simulation_length + 1)
    entity_position_y = np.zeros(simulation_length * 9 + 9, dtype=np.longdouble).reshape(9, simulation_length + 1)

    entity_velocity_x = np.zeros(simulation_length * 9 + 9, dtype=np.longdouble).reshape(9, simulation_length + 1)
    entity_velocity_y = np.zeros(simulation_length * 9 + 9, dtype=np.longdouble).reshape(9, simulation_length + 1)

    entity_mass = np.array([1988500e24, 0.33011e24, 4.8675e24, 5.9723e24, 0.64171e24, 1898.19e24, 568.34e24, 86.813e24,
                            102.413e24], dtype=np.longdouble)

    # Set up the initial position and velocity data
    entity_position_x[:, 0] = np.array([0, 69.82e9, 108.94e9, 152.10e9, 249.23e9, 816.62e9, 1514.50e9,
                                        3003.62e9, 4545.67e9], dtype=np.longdouble)

    entity_velocity_y[:, 0] = np.array([0, 38.86e3, 34.79e3, 29.29e3, 21.97e3, 12.44e3, 9.09e3,
                                        6.49e3, 5.37e3], dtype=np.longdouble)

    print("Simulating...")
    print("There are {} simulation steps in total".format(simulation_length))
    a = time.time()
    for i in range(0, simulation_length):

        # Did not calculate the motion of the Sun since Sun will be in the center of the graph
        # Perform RK4 integration step for all planets
        for p in range(1, 9):
            k1_xa, k1_ya = calc_acc(entity_position_x[p, i], entity_position_x[:, i], entity_position_y[p, i],
                                    entity_position_y[:, i], p, entity_mass)

            k1_xv = entity_velocity_x[p, i]
            k1_yv = entity_velocity_y[p, i]

            k2_xa, k2_ya = calc_acc(entity_position_x[p, i] + (dt / 2) * k1_xv, entity_position_x[:, i],
                                    entity_position_y[p, i] + (dt / 2) * k1_yv,
                                    entity_position_y[:, i], p, entity_mass)

            k2_xv = entity_velocity_x[p, i] + (dt / 2) * k1_xa
            k2_yv = entity_velocity_y[p, i] + (dt / 2) * k1_ya

            k3_xa, k3_ya = calc_acc(entity_position_x[p, i] + (dt / 2) * k2_xv, entity_position_x[:, i],
                                    entity_position_y[p, i] + (dt / 2) * k2_yv,
                                    entity_position_y[:, i], p, entity_mass)

            k3_xv = entity_velocity_x[p, i] + (dt / 2) * k2_xa
            k3_yv = entity_velocity_y[p, i] + (dt / 2) * k2_ya

            k4_xa, k4_ya = calc_acc(entity_position_x[p, i] + dt * k3_xv, entity_position_x[:, i],
                                    entity_position_y[p, i] + dt * k3_yv,
                                    entity_position_y[:, i], p, entity_mass)

            k4_xv = entity_velocity_x[p, i] + dt * k3_xa
            k4_yv = entity_velocity_y[p, i] + dt * k3_ya

            entity_position_x[p, i + 1] = entity_position_x[p, i] + (dt / 6) * (k1_xv + 2 * k2_xv + 2 * k3_xv + k4_xv)
            entity_position_y[p, i + 1] = entity_position_y[p, i] + (dt / 6) * (k1_yv + 2 * k2_yv + 2 * k3_yv + k4_yv)

            entity_velocity_x[p, i + 1] = k1_xv + (dt / 6) * (k1_xa + 2 * k2_xa + 2 * k3_xa + k4_xa)
            entity_velocity_y[p, i + 1] = k1_yv + (dt / 6) * (k1_ya + 2 * k2_ya + 2 * k3_ya + k4_ya)

    c = time.time()
    print("RK4 simulation took:", c - a, "s to finish")

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

    diff = calculate_energy_diff(entity_position_x, entity_position_y, entity_velocity_x, entity_velocity_y,
                                 entity_mass)
    print("The energy difference for RK4 method is:{}%".format(diff))
    plt.legend()
    plt.title("RK4 Planet Trajectory simulation for {} years".format(simulation_time / 3.154e+7))
    plt.xlabel("distance, unit: m")
    plt.ylabel("distance, unit: m")
    plt.savefig("RK4 {}.png".format(simulation_time / 3.154e+7))


if __name__ == "__main__":
    print("=================================Start dt=3600 1 years=============================")
    print("-------------------Verlet-----------------------")
    simulation_verlet(3.154e+7, 3600)
    print("-------------------RK4-----------------------")
    simulation_Runge_kutta_4rd(3.154e+7, 3600)
    print("=================================Start dt=1800 1 years=============================")
    print("-------------------Verlet-----------------------")
    simulation_verlet(3.154e+7, 1800)
    print("-------------------RK4-----------------------")
    simulation_Runge_kutta_4rd(3.154e+7, 1800)

    print("=================================Start dt=3600 17 years============================")
    print("-------------------Verlet-----------------------")
    simulation_verlet(3.154e+7 * 17, 3600)
    print("-------------------RK4-----------------------")
    simulation_Runge_kutta_4rd(3.154e+7 * 17, 3600)
    print("=================================Start dt=1800 17 years============================")
    print("-------------------Verlet-----------------------")
    simulation_verlet(3.154e+7 * 17, 1800)
    print("-------------------RK4-----------------------")
    simulation_Runge_kutta_4rd(3.154e+7 * 17, 1800)

    print("=================================Start dt=3600 165 years ==========================")
    print("-------------------Verlet-----------------------")
    simulation_verlet(3.154e+7 * 165, 3600)
    print("-------------------RK4-----------------------")
    simulation_Runge_kutta_4rd(3.154e+7 * 165, 3600)
    print("=================================Start dt=1800 165 years ==========================")
    print("-------------------Verlet-----------------------")
    simulation_verlet(3.154e+7 * 165, 1800)
    print("-------------------RK4-----------------------")
    simulation_Runge_kutta_4rd(3.154e+7 * 165, 1800)
