import numpy as np
import json
import os


class PoincareVectors:
    """
    class to initialize, store, load and return quickly embedding vectors
    """
    def __init__(self,
                 index2node,
                 node2index,
                 embedding_dimension=100,
                 low_bound=-0.0001,
                 high_bound=0.0001):
        self.index2node = index2node
        self.node2index = node2index
        self.number_of_nodes = len(index2node)
        self.embedding_dimension = embedding_dimension
        self.low_bound = low_bound
        self.high_bound = high_bound

        self.vectors = None

        self.build_vectors()

        return

    def build_vectors(self):
        """
        instantiate embedding vectors
        :return: nothing
        """
        self.vectors = dict(
            zip(self.index2node.keys(),
                map(lambda x: np.random.uniform(low=self.low_bound,
                                                high=self.high_bound,
                                                size=(self.embedding_dimension, 1)),
                    self.index2node.keys())
                )
        )

    def get_vectors(self, list_of_index):
        """
        return vectors according to a list of index
        :param list_of_index: list of index fron which we want the embedding vectors
        :return: a list of vectors of dimension [len(list_of_index), embedding_dimension, 1]
        """

        vectors = map(self.vectors.get, list_of_index)

        vectors = np.stack(vectors, axis=0)
        return vectors

    def save_data(self, path, postfix=''):
        """
        saves the embedding vectors in a specified folder in a json file
        :param path: path to a folder - if it does not exist, it will be created with a subfolder 'embeddings'
        :param postfix: postfix to add at the end of the name of the json file
        :return: nothing
        """
        try:
            os.stat(os.path.join(path, 'embeddings'))
        except FileNotFoundError:
            print('creating folder')
            try:
                os.mkdir(path=path)
            except FileExistsError:
                pass
            finally:
                os.mkdir(path=os.path.join(path, 'embeddings'))

        print('formatting embeddings ...')
        embeddings = dict(
            zip(
                self.vectors.keys(),
                map(lambda x: list(self.vectors.get(x).ravel()),
                    self.vectors.keys())
            )
        )
        with open(os.path.join(path, 'embeddings', 'embedding' + postfix + '.json'),
                  'w', encoding='utf-8') as embedding_file:
            json.dump(obj=embeddings, fp=embedding_file)

        print('embeddings saved.')
        return

    def load_data(self, path, postfix):
        """
        loads embedding vectors from a json file in <path>/embeddings/embedding<postfix>.json
        :param path: path to the folder containing the subfolder embeddings
        :param postfix: postfix completing the name of the json file containing the embedding vectors
        :return: nothing
        """
        with open(os.path.join(path, 'embeddings', 'embedding' + postfix + '.json'),
                  'r', encoding='utf-8') as embedding_file:
            embeddings = json.load(fp=embedding_file)

        print('formatting embeddings ...')
        embeddings = dict(
            zip(
                embeddings.keys(),
                map(lambda x: np.array(x).reshape(self.embedding_dimension, 1),
                    embeddings.values())
            )
        )

        self.vectors = embeddings
        self.number_of_nodes = len(embeddings)
        self.node2index = dict(zip(self.vectors.keys(), range(self.number_of_nodes)))
        self.index2node = {index: node for node, index in self.node2index.items()}
        print('embeddings loaded.')
        return


if __name__ == '__main__':
    keyed_vectors = PoincareVectors(index2node=dict(zip(range(100), range(100))), node2index=None)

    output = keyed_vectors.get_vectors([1, 2, 3])

    # print(output)

    print(output.shape)

    keyed_vectors.save_data(path='test1', postfix='_test')
    keyed_vectors.load_data(path='test1', postfix='_test')
