import configparser
import os
import subprocess
import sys
from urllib.request import urlretrieve

CURDIR = os.path.dirname(os.path.abspath(__file__))
CONFIGPATH = os.path.join(CURDIR, os.pardir, 'config.ini')
config = configparser.ConfigParser()
config.read(CONFIGPATH)

FILEURL = config['DATA']['FILEURL']
FILEPATH = config['DATA']['FILEPATH']
EXTRACTDIR = config['DATA']['TEXTDIR']

def extract():
    subprocess.call(['python3', 
                    os.path.join(CURDIR, os.pardir,
                                 'wikiextractor', 'WikiExtractor.py'), 
                    FILEPATH, "-o={}".format(EXTRACTDIR)])


def main():
    download()
    extract()


if __name__ == "__main__":
    main()