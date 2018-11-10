import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


class PoincareModel:
    def __init__(self,
                 graph,
                 embedding_size=100,
                 negative_sample_size=10,
                 batch_size=10,
                 epsilon=10e-5,
                 learning_rate=.001,
                 learning_rate_burn_in=0.1,
                 epochs_burn_in=10):

        # feeding the graph
        self.graph = graph

        # parameters of the embedding
        self.embedding_size = embedding_size
        self.negative_sample_size = negative_sample_size
        self.batch_size = batch_size
        self.epsilon = epsilon

        # Learning rate
        self.normal_learning_rate = learning_rate

        # burn in
        self.learning_rate_burn_in = learning_rate_burn_in
        self.learning_rate = learning_rate_burn_in
        self.epochs_burn_in = epochs_burn_in
        self.burn_in_done = False

        # building a tensorflow session
        self.session = tf.Session()

        # building a tensorflow model
        self._build_model()
        return

    def _build_model(self):
        # input for the concerned node: u
        self.input_node = tf.placeholder(dtype=tf.float64, shape=[1, self.embedding_size, None])
        # input for the link at the other side of the edge: v
        self.input_linked_node = tf.placeholder(dtype=tf.float64, shape=[1, self.embedding_size, None])
        # input for negative samples: N(u)
        self.input_negative_nodes = tf.placeholder(dtype=tf.float64,
                                                   shape=[self.negative_sample_size, self.embedding_size, None])

        # repeat the input nodes to compute all distances at once
        self.input_nodes_tiling = tf.tile(input=self.input_node, multiples=[self.negative_sample_size + 1, 1, 1])
        # denominator for the loss function: N(u) U v
        self.denominator = tf.concat(values=[self.input_negative_nodes, self.input_linked_node], axis=0)
        # denominator norm
        self.denominator_norm = hyperbolic_distance(self.input_nodes_tiling, self.denominator)
        # total denominator norm
        self.total_denominator_norm = tf.reduce_sum(input_tensor=tf.exp(-self.denominator_norm), axis=0)
        # numerator norm
        self.numerator_norm = tf.reduce_sum(tf.exp(-hyperbolic_distance(self.input_node, self.input_linked_node)),
                                            axis=0)

        # loss function
        self.loss = tf.log(self.numerator_norm / self.total_denominator_norm)

        # gradient loss
        self.loss_gradient = tf.gradients(ys=self.loss, xs=self.input_node)[0]

        # computing the update before projection on the unit ball
        self.not_projected_update = self.input_node - self.learning_rate * tf.square(1 - tf.square(
            tf.norm(tensor=self.input_node, ord='euclidean', axis=1)
        )) * self.loss_gradient

        # computing and resizing the norm
        norm = tf.reshape(tf.norm(self.not_projected_update, ord='euclidean', axis=1), shape=[1, 1, -1])
        norm = tf.tile(norm, multiples=[1, self.embedding_size, 1])

        # projecting the vector onto the unit ball
        self.update = tf.where(condition=norm >= 1,
                               x=self.not_projected_update / norm - self.epsilon,
                               y=self.not_projected_update)
        return

    def _update_learning_rate(self, learning_rate):
        # computing the update before projection on the unit ball
        self.learning_rate = learning_rate
        self.not_projected_update = self.input_node - self.learning_rate * tf.square(1 - tf.square(
            tf.norm(tensor=self.input_node, ord='euclidean', axis=1)
        )) * self.loss_gradient

        # computing and resizing the norm
        norm = tf.reshape(tf.norm(self.not_projected_update, ord='euclidean', axis=1), shape=[1, 1, -1])
        norm = tf.tile(norm, multiples=[1, self.embedding_size, 1])

        # projecting the vector onto the unit ball
        self.update = tf.where(condition=norm >= 1,
                               x=self.not_projected_update / norm - self.epsilon,
                               y=self.not_projected_update)
        return

    def _train(self, input_node, input_linked_node, input_negative_nodes, verbose=True):

        loss, grad, update = self.session.run(
            fetches=[self.loss, self.loss_gradient, self.update],
            feed_dict={
                self.input_node: input_node,
                self.input_linked_node: input_linked_node,
                self.input_negative_nodes:  input_negative_nodes,
            }
        )
        if verbose:
            print(sum(loss))

        return loss, update


def hyperbolic_distance(u, v):
    """
    computes the hyperbolic distance between u and v
    :param u: a tensor of size [1, embedding_size, batch_size]
    :param v: a tensor of size [1, embedding_size, batch_size]
    :return: a tensor of size [1, batch_size] containing the distances
    """
    difference_norm = tf.norm(
        tf.subtract(x=u, y=v),
        ord='euclidean',
        axis=1)
    norm_u = tf.norm(tensor=u, ord='euclidean', axis=1)
    norm_v = tf.norm(tensor=v, ord='euclidean', axis=1)

    content = 1 + 2 * tf.square(difference_norm) / (1 - tf.square(norm_u)) / (1 - tf.square(norm_v))

    return tf.math.acosh(content)


if __name__ == '__main__':

    model = PoincareModel(graph=None)

    print(model.input_node.shape)
    embedding_dimension = 100
    negative_samples = 10
    batch_size_ = 100

    input_nodes = np.random.uniform(low=-0.001,
                                    high=0.001,
                                    size=(1, embedding_dimension, batch_size_))
    input_linked_nodes = np.random.uniform(low=-0.001,
                                           high=0.001,
                                           size=(1, embedding_dimension, batch_size_))
    negative_nodes_inputs = np.random.uniform(low=-0.001,
                                              high=0.001,
                                              size=(negative_samples, embedding_dimension, batch_size_))

    print(input_nodes.shape)
    print(input_linked_nodes.shape)
    print(negative_nodes_inputs.shape)

    print('training')

    losses = []

    for i in range(100):
        loss_, update_ = model._train(input_node=input_nodes,
                                      input_linked_node=input_linked_nodes,
                                      input_negative_nodes=negative_nodes_inputs,
                                      verbose=False)
        input_nodes = update_
        losses.append(np.mean(loss_))
        model._update_learning_rate(learning_rate=0.0001)

    plt.plot(losses)
    plt.show()
