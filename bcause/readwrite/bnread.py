import logging
import os

from pgmpy.readwrite import BIFReader, XMLBIFReader

from conversion.pgmpy import toBCauseBNet
from util.assertions import assert_file_exists


def __read(reader, file):
    assert_file_exists(file)
    logging.info(f"Reading model in {reader.__name__.replace('Reader', '')} format from {os.path.abspath(file)}")
    model =  toBCauseBNet(reader(file).get_model())
    logging.debug(f"Loaded {model}")
    return model

def fromBIF(file):
    return __read(BIFReader, file)

def fromXMLBIF(file):
    return __read(XMLBIFReader, file)
