from gensim.models.poincare import PoincareModel
from downloads_utils import format_data
import matplotlib.pyplot as plt
# import numpy as np

from autograd import numpy as np


path = 'ca-GrQc.txt.gz'


def projection_on_unit_ball(vector, epsilon=.00001):
    norm = np.linalg.norm(vector, axis=0)
    vector_ = vector.copy()

    vector_[:, norm >= 1] = vector_[:, norm >= 1] / norm[norm >= 1] - epsilon

    return vector_


def square(vector):
    return np.power(vector, 2)


def euclidean_norm_squared(vector):
    return square(np.linalg.norm(vector, axis=0))


def derive(vector_t, vector_x):
    norm_t2 = euclidean_norm_squared(vector_t)
    norm_x2 = euclidean_norm_squared(vector_x)

    alpha = 1 - norm_t2
    beta  = 1 - norm_x2

    gamma = 1 + 2/(alpha*beta)*euclidean_norm_squared(vector_t-vector_x)

    derivative = 4/(beta * square(alpha)*np.sqrt(square(gamma)-1)) * ((norm_x2 - 2 * np.dot(vector_x, vector_t) + 1) * vector_t - vector_x * alpha)
    return derivative


def geometric_distance(vector_u, vector_v):
    temp = 1 + 2 * euclidean_norm_squared(vector_u-vector_v)
    print(temp)
    temp = temp / ((1 - euclidean_norm_squared(vector_u))*(1 - euclidean_norm_squared(vector_v)))
    print(temp)
    return np.arccosh(temp)

if __name__ == "__main__":

    points = np.random.uniform(low=-1, high=1, size=(10000, 2))

    normed_points = projection_on_unit_ball(points)

    x = np.linspace(start=0, stop=2 * np.pi, num=1000)
    y = np.sin(x)
    x = np.cos(x)

    # plotting the results
    plt.figure(figsize=(8, 8))
    plt.plot(points[:, 0], points[:, 1], linestyle='', marker='+', color='b', alpha=.3)
    plt.plot(normed_points[:, 0], normed_points[:, 1], linestyle='', marker='+', color='r', alpha=.3)
    plt.plot(x, y, color='y')
    plt.show()

    a = np.array([[.1, .22], [.1, .2]])
    b = np.array([[0, .5], [0, 0]])

    print(derive(a, b))

    c = np.random.uniform(size=(1, 2)).T
    d = np.random.uniform(size=(1, 2)).T

    print(geometric_distance(projection_on_unit_ball(c), projection_on_unit_ball(d)))

    x = np.array([(i-500.)/500. for i in range(1001)])
    y = x
    x, y = np.meshgrid(x, y)
    z = np.vstack([x.ravel(), y.ravel()])
    print(z.shape)
    print(x.shape)
    print(y.shape)
    z_proj = projection_on_unit_ball(z)
    print(z_proj.shape)
    distances = geometric_distance(z_proj, np.zeros(shape=(2, x.shape[0]*x.shape[1])))

    distances = distances.reshape(x.shape[0], x.shape[1])
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt
    from matplotlib import cm

    fig = plt.figure()
    ax = fig.gca(projection='3d')

    surf = ax.plot_surface(x, y, distances,  cmap=cm.coolwarm, alpha=.3,
                           linewidth=0, antialiased=False)

    # Customize the z axis.
    ax.set_zlim(0, 10)
    # ax.zaxis.set_major_locator(LinearLocator(10))
    # ax.zaxis.set_major_formatter(FormatStrFormatter('%.02f'))

    # Add a color bar which maps values to colors.
    fig.colorbar(surf, shrink=0.5, aspect=5)

    plt.show()