import numpy as np
import matplotlib.pyplot as plt


def plot_hyperboloid_model_2d():
    x = np.linspace(start=1, stop=2, num=1000)
    y1 = np.sqrt(np.power(x, 2)-1)
    y2 = -np.sqrt(np.power(x, 2)-1)

    plt.figure(figsize=(8, 8))
    plt.plot(y1, x, color='b')
    plt.plot(y2, x, color='b')
    first_level = [np.sqrt(np.power(1.2, 2)-1), -np.sqrt(np.power(1.2, 2)-1)]
    second_level = [np.sqrt(np.power(1.4, 2)-1),
                    np.sqrt(np.power(1.4, 2)-1) - 2 * np.sqrt(np.power(1.2, 2)-1),
                    2 * np.sqrt(np.power(1.2, 2) - 1) - np.sqrt(np.power(1.4, 2) - 1),
                    -np.sqrt(np.power(1.4, 2)-1)]

    plt.scatter([0, first_level[0], first_level[-1], second_level[0], second_level[-1], second_level[1], second_level[2]],
                [1, 1.2, 1.2, 1.4, 1.4, 1.4, 1.4], color='r')
    plt.plot([0, 0], [1, 1.2], color='k')
    plt.plot([first_level[0], first_level[0]], [1.2, 1.4], color='k')
    plt.plot([first_level[-1], first_level[-1]], [1.2, 1.4], color='k')
    plt.plot([first_level[-1], second_level[-1]], [1.4, 1.4], color='k')
    plt.plot([first_level[0], second_level[0]], [1.4, 1.4], color='k')
    plt.plot([first_level[-1], second_level[1]], [1.4, 1.4], color='k')
    plt.plot([first_level[0], second_level[2]], [1.4, 1.4], color='k')
    plt.plot([np.sqrt(np.power(1.2, 2)-1), -np.sqrt(np.power(1.2, 2)-1)], [1.2, 1.2], color='k')
    plt.axvline(0, color='k', linestyle='--', alpha=.3)
    plt.axhline(0, color='k', linestyle='--', alpha=.3)
    plt.xlim(-2, 2)
    plt.ylim(0, 2)
    plt.savefig('tree_in_hyperbolic_space.png')
    plt.show()


if __name__ == '__main__':
    plot_hyperboloid_model_2d()