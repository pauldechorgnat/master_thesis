import wget
import gzip
import os
import re
import numpy as np
import random

# defining the URLs for the data sets
URLS = {
    "GrQc": "https://snap.stanford.edu/data/ca-GrQc.txt.gz",
    "HepPh": "https://snap.stanford.edu/data/ca-HepPh.txt.gz",
    "AstroPh": "https://snap.stanford.edu/data/ca-AstroPh.txt.gz",
    "CondMat": "https://snap.stanford.edu/data/ca-CondMat.txt.gz"
}


def download_and_extract(url):
    # name of the file to download
    file_name = url.split('/')[-1]

    # if the file is not already downloaded, do it
    if file_name not in os.listdir("."):
        wget.download(url=url)

    # extracting data
    lines = []
    with gzip.open(file_name, "r") as file:
        for line in file:
            lines.append(line)
    return lines


def import_data(path):
    counter = 0
    edges = []
    regex = re.compile(b'[0-9]+')
    with gzip.open(path, 'rb') as file:
        for line in file:
            if counter > 0:
                edge = regex.findall(line)
                if len(edge) == 2:
                    edges.append(tuple([str(edge[0]), str(edge[1])]))
            counter += 1

    return edges


def quick_sample(set_of_nodes, probabilities):
    """
    quickly generate a random element from a set of nodes
    :param set_of_nodes: set of nodes from which we want to return a random sample
    :param probabilities: drawing probabilities in ascending order
    :return: a random node index
    """
    return list(set_of_nodes)[np.searchsorted(probabilities, random.random())]


if __name__ == "__main__":
    data_ = import_data(path='ca-GrQc.txt.gz')
    print(data_)
