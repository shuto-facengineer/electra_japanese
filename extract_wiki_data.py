import os
import subprocess
import sys

CURDIR = os.path.dirname(os.path.abspath(__file__))


FILEURL = "https://dumps.wikimedia.org/jawiki/latest/jawiki-latest-pages-articles-multistream.xml.bz2"
FILEPATH = "data/jawiki-latest-pages-articles-multistream.xml.bz2"
EXTRACTDIR = "data/wiki/"


def extract():
    subprocess.call(['python3', 
                    os.path.join(CURDIR, os.pardir,
                                 'wikiextractor', 'WikiExtractor.py'), 
                    FILEPATH, "-o={}".format(EXTRACTDIR)])


def main():
    extract()


if __name__ == "__main__":
    main()