import wget
import gzip
import os
import re

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


def format_data(file_path):
    regex = re.compile(pattern=b"[0-9]+\t[0-9]+")
    with gzip.open(file_path, "rb") as file:
        file_content = file.read()
        edges = [tuple(edge.split(b"\t")) for edge in regex.findall(string=file_content)]
    return iter(edges)


def import_data(path):
    counter = 0
    edges = []
    regex = re.compile(b'[0-9]+')
    with gzip.open(path, 'rb') as file:
        for line in file:
            if counter > 0:
                edge = regex.findall(line)
                if len(edge) == 2:
                    edges.append(tuple(edge))
            counter += 1

    return edges

if __name__ == "__main__":

    data_ = format_data(file_path='ca-GrQc.txt.gz')
    print(data_)