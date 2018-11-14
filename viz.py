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


def plot_2d_word_embedding():

    plt.figure(figsize=(8, 8))
    plt.xlim(-1.01, 1.01)
    plt.ylim(-1.01, 1.01)
    plt.arrow(x=-1, y=0, dx=2, dy=0,
              length_includes_head=True, head_width=0.05, head_length=0.1, color='r', label='gender')
    plt.arrow(x=0, y=-1, dx=0, dy=2,
              length_includes_head=True, head_width=0.05, head_length=0.1, color='r', label='royalty')
    plt.axvline(0, color='r', alpha=0.9)
    plt.arrow(x=0, y=0, dx=0.75, dy=0.75, label='KING', alpha=.5,
              length_includes_head=True, head_width=0.05, head_length=0.1, color='k')
    plt.arrow(x=0, y=0, dx=-0.75, dy=0.75, label='QUEEN',alpha=.5,
              length_includes_head=True, head_width=0.05, head_length=0.1, color='k')
    plt.arrow(x=0, y=0, dx=1, dy=0, label='MAN',alpha=.5,
              length_includes_head=True, head_width=0.05, head_length=0.1, color='k')
    plt.arrow(x=0, y=0, dx=-1, dy=0, label='MAN',alpha=.5,
              length_includes_head=True, head_width=0.05, head_length=0.1, color='k')
    plt.annotate('Royalty', [0, .5], color='red')
    plt.annotate('Masculinity', [.5, 0], color='red')
    plt.annotate('MAN', xy=[0.75, 0.1])
    plt.annotate('WOMAN', xy=[-0.75, 0.1])
    plt.annotate('KING', xy=[0.75, 0.75])
    plt.annotate('QUEEN', xy=[-0.75, 0.75])

    plt.axis('off')
    plt.savefig('2d_embedding_king_man_queen_woman.png')
    plt.show()


if __name__ == '__main__':
    # plot_hyperboloid_model_2d()
    plot_2d_word_embedding()
