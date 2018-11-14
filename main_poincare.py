from graph import Graph
from loss_function import PoincareTrainer
from keyed_vectors import PoincareVectors
import numpy as np
import random


class PoincareModel:
    def __init__(self,
                 path_to_folder,
                 path_to_data,
                 embedding_dimension=100,
                 low_bound=-0.0001,
                 high_bound=0.0001,
                 negative_sample_size=10):
        self.path_to_folder = path_to_folder
        self.path_to_data = path_to_data
        self.graph = Graph()
        self.graph.import_data(path=path_to_data, limit=1000)
        self.embedding_dimension = embedding_dimension
        self.negative_sample_size = negative_sample_size
        self.low_bound = low_bound
        self.high_bound = high_bound
        self.vectors = PoincareVectors(index2node=self.graph.index2node,
                                       node2index=self.graph.node2index,
                                       embedding_dimension=embedding_dimension,
                                       low_bound=low_bound,
                                       high_bound=high_bound
                                       )
        self.trainer = PoincareTrainer(embedding_size=embedding_dimension,
                                       negative_sample_size=negative_sample_size)

        return

    def _generate_batch_index(self, batch_size=10):
        # for each edge, we generate a sample of random samples
        # shuffling edges
        edges = list(self.graph.edges)
        random.shuffle(x=edges)
        # negative sampling
        negative_samples = list(
            map(lambda edge: self.graph.negative_sample(edge[0], size=self.negative_sample_size),
                edges)
        )
        edges_with_negative_samples = list(zip(edges, negative_samples))
        batch_edges = map(lambda i: edges_with_negative_samples[i * batch_size: (i + 1)*batch_size],
                          range(len(edges)//batch_size))

        return list(batch_edges)

    def _generate_batch_vectors(self, batch_edges, batch_number=0):
        batch = batch_edges[batch_number]
        batch_vectors_node = self.vectors.get_vectors([edge[0] for edge, _ in batch_edges[batch_number]])
        batch_vectors_linked_node = self.vectors.get_vectors([edge[1] for edge, _ in batch_edges[batch_number]])
        batch_vectors_negative = [self.vectors.get_vectors(negative).T for _, negative in batch_edges[batch_number]]

        return np.array(batch_vectors_node), np.array(batch_vectors_linked_node), np.array(batch_vectors_negative)


if __name__ == '__main__':
    model = PoincareModel(path_to_folder='test', path_to_data='ca-GrQc.txt.gz')
    obj = model._generate_batch_index()
    nodes, linked, negative = model._generate_batch_vectors(obj)

    print(nodes.shape)
    print(linked.shape)
    print(negative.shape)
