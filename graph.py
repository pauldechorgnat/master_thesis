from autograd import numpy as np
from downloads_utils import format_data
from tqdm import tqdm
from downloads_utils import import_data
import random


class Graph:
    def __init__(self):
        self.nodes = set()
        self.edges = set()
        self.neighbours = dict()
        self.negative = dict()
        self.node2index = dict()
        self.index2node = dict()
        self.number_of_nodes = 0
        return

    def load_data(self, path):
        edges_raw = import_data(path)[:1000]

        counter = -1

        for node_a, node_b in tqdm(edges_raw):

            if node_a not in self.node2index:
                counter += 1
                self.node2index[node_a] = counter

            if node_b not in self.node2index:
                counter += 1
                self.node2index[node_b] = counter

            node_a = self.node2index[node_a]
            node_b = self.node2index[node_b]


            # keeping the list of nodes
            self.nodes.update([node_a])
            self.nodes.update([node_b])

            # for each node, we will keep its neighbour
            self.neighbours[node_a] = self.neighbours.get(node_a, []) + [node_b]
            self.neighbours[node_b] = self.neighbours.get(node_b, []) + [node_a]

            # keeping the list of edges
            self.edges.update([(node_a, node_b)])
            self.edges.update([(node_b, node_a)])

        self.number_of_nodes = counter + 1

        # changing the values of neighbours into a set
        for node in self.neighbours:
            self.neighbours[node] = set(self.neighbours[node])

        # creating the negative samples
        for index_node_a, node_a in tqdm(enumerate(self.nodes)):
            for node_b in self.nodes:  # list(self.nodes)[index_node_a+1:]:
                if node_b not in self.neighbours[node_a]:
                    self.negative[node_a] = self.negative.get(node_a, []) + [node_b]
                    self.negative[node_b] = self.negative.get(node_b, []) + [node_a]

        for node in self.negative:
            self.negative[node] = set(self.negative[node])

        self.index2node = {index: node for node, index in self.node2index.items()}
        pass

    def negative_sample(self, node, size=10):
        number_of_total_negative_samples = len(self.negative[node])
        probabilities = [i/number_of_total_negative_samples for i in range(1, number_of_total_negative_samples+1)]

        outputs = [list(self.negative[node])[np.searchsorted(probabilities, random.random())] for _ in range(size)]
        return outputs


class PoincareTrainer:
    def __init__(self):
        pass







if __name__ == "__main__":
    np.random.uniform()
    graph = Graph()
    graph.load_data('ca-GrQc.txt.gz')
    print(len(graph.nodes))
    print(graph.number_of_nodes)
    print(graph.negative.keys())
    graph.negative_sample(node=0, size=10)
