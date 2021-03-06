from tqdm import tqdm
from utils import import_data, quick_sample
import os
import json


class Graph:
    """
    class to store nodes, edges, neighbours and unlinked nodes
    """
    def __init__(self):
        self.nodes = set()
        self.edges = set()
        self.neighbours = dict()
        self.negative = dict()
        self.node2index = dict()
        self.index2node = dict()
        self.number_of_nodes = 0
        return

    def import_data(self, path, limit=None):
        """
        Imports graph data from a text.gz file
        Data is supposed to be in the form of a list of edges:
        - one edge per line
        - nodes are separated by a \t
        - the first line is just a header

        :param path: path to the gzip file
        :param limit: limit the number of edges to keep - default is None -> no limit
        :return: nothing
        """
        edges_raw = import_data(path)[:limit]

        counter = -1

        for node_a, node_b in tqdm(edges_raw):

            if node_a not in self.node2index:
                counter += 1
                self.node2index[node_a] = counter

            if node_b not in self.node2index:
                counter += 1
                self.node2index[node_b] = counter

            # changing nodes into node index
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

        # changing the values of neighbours list into a set
        self.neighbours = dict(zip(self.neighbours.keys(), map(set, self.neighbours.values())))

        # creating a dictionary for negative samples: the values are a set of all the nodes
        self.negative = dict()  # zip(self.nodes, [set(self.nodes) for _ in range(self.number_of_nodes)]))

        # removing neighbours from negative samples
        nodes = set(self.nodes)
        del edges_raw

        for node in tqdm(self.nodes):
            self.negative[node] = nodes.difference(self.neighbours[node])
            # self.negative[node].remove(node)

        # creating an index to node dictionary
        self.index2node = {index: node for node, index in self.node2index.items()}
        return

    def save_data(self, path):
        """
        saves the data into a folder
        :param path: name of a folder where to put the data - if the folder does not exist, it will be created
        :return: nothing
        """
        # checking the existence of the folder
        try:
            os.stat(path=path)
        except FileNotFoundError:
            print('creating folder {}'.format(path))
            os.mkdir(path=path)

        # saving indexes
        print('saving indexes ...')
        with open(os.path.join(path, 'index2node.json'), 'w', encoding='utf-8') as index2node_file:
            json.dump(obj=self.index2node, fp=index2node_file)
        with open(os.path.join(path, 'node2index.json'), 'w', encoding='utf-8') as node2index_file:
            json.dump(obj=self.node2index, fp=node2index_file)

        # saving node list
        print('saving nodes list ...')
        with open(os.path.join(path, 'nodes.txt'), 'w', encoding='utf-8') as nodes_file:
            nodes_file.write('\t'.join([str(node) for node in self.nodes]))

        # saving edges list
        print('saving edges list ...')
        with open(os.path.join(path, 'edges.txt'), 'w', encoding='utf-8') as edges_file:
            edges_file.write('\n'.join([str(node1)+'\t'+str(node2) for node1, node2 in self.edges]))

        # saving neighbours
        print('saving neighbours ...')
        neighbours = self.neighbours.copy()
        for n in neighbours.keys():
            neighbours[n] = list(neighbours[n])
        with open(os.path.join(path, 'neighbours.json'), 'w', encoding='utf-8') as neighbours_file:
            json.dump(obj=neighbours, fp=neighbours_file)

        # saving negative nodes
        print('saving negative nodes ...')
        negative = self.negative.copy()
        for n in negative.keys():
            negative[n] = list(negative[n])
        with open(os.path.join(path, 'negative_nodes.json'), 'w', encoding='utf-8') as negative_file:
            json.dump(obj=negative, fp=negative_file)

        print('graph saved.')
        return

    def load_data(self, path):
        """
        loading data from a specified folder
        :param path: path to the folder where the data is supposed to be kept
        :return: nothing
        """
        # loading indexes
        print('loading indexes ...')
        with open(os.path.join(path, 'index2node.json'), 'r', encoding='utf-8') as index2node_file:
            index2node = json.load(fp=index2node_file)
        with open(os.path.join(path, 'node2index.json'), 'r', encoding='utf-8') as node2index_file:
            node2index = json.load(fp=node2index_file)
        self.index2node = index2node
        self.node2index = node2index

        # loading nodes
        print('loading nodes ...')
        with open(os.path.join(path, 'nodes.txt'), 'r', encoding='utf-8') as nodes_file:
            nodes = [int(node) for node in nodes_file.read().split('\t')]
        self.nodes = nodes
        self.number_of_nodes = len(nodes)

        # loading edges
        print('loading edges ...')
        edges = []
        with open(os.path.join(path, 'edges.txt'), 'r', encoding='utf-8') as edges_file:
            for line in edges_file:
                edge = line.split('\t')
                edges.append((int(edge[0]), int(edge[1])))
        self.edges = edges

        # loading neighbours
        print('loading neighbours ...')
        with open(os.path.join(path, 'neighbours.json'), 'r', encoding='utf-8') as neighbours_file:
            neighbours = json.load(fp=neighbours_file)
        neighbours_ = dict()
        for n in neighbours.keys():
            neighbours_[int(n)] = set(map(int, neighbours[n]))
        self.neighbours = neighbours_

        # loading negative nodes
        print('loading negative nodes ...')
        with open(os.path.join(path, 'negative_nodes.json'), 'r', encoding='utf-8') as negative_file:
            negative = json.load(fp=negative_file)
        negative_ = dict()
        for n in negative:
            negative_[int(n)] = set(map(int, negative[n]))
        self.negative = negative_

        print('data loaded.')
        return

    def negative_sample(self, node, size=10):
        """
        given a node, returns a list of nodes that are not linked
        :param node: node from which we want to return a negative sample
        :param size: number of nodes to return
        :return: a list of index corresponding to the negative samples
        """
        number_of_total_negative_samples = len(self.negative[node])
        probabilities = [i/number_of_total_negative_samples for i in range(1, number_of_total_negative_samples+1)]

        outputs = map(lambda x: quick_sample(set_of_nodes=self.negative[node],
                                             probabilities=probabilities),
                      range(size))
        return list(outputs)


if __name__ == "__main__":

    graph = Graph()
    graph.import_data('ca-CondMat.txt.gz', limit=None)
    # print(len(graph.nodes))
    # print(graph.number_of_nodes)
    # print(graph.negative.keys())
    graph.negative_sample(node=0, size=10)
    graph.save_data(path='test1')
    graph.load_data(path='test1')
    # node = 0
    # print(node in graph.neighbours[0])

    # print(graph.edges)
    # input()
    # print(graph.nodes)
    # input()
    # print(graph.neighbours)
    # input()
    # print(graph.negative[0])
