import wget
import gzip
import os


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


if __name__ == "__main__":
    for data_set_name in URLS.keys():
        print(data_set_name)
        _ = download_and_extract(URLS[data_set_name])
