import logging
import os

from pgmpy.readwrite import BIFReader, XMLBIFReader

from bcause.conversion.pgmpy import toBCauseBNet
from bcause.util.assertions import assert_file_exists


def __read(reader, file, vtype):
    assert_file_exists(file)
    logging.info(f"Reading model in {reader.__name__.replace('Reader', '')} format from {os.path.abspath(file)}")
    model =  toBCauseBNet(reader(file).get_model(), vtype)
    logging.debug(f"Loaded {model}")
    return model

def fromBIF(file, vtype="numpy"):
    return __read(BIFReader, file, vtype)

def fromXMLBIF(file, vtype="numpy"):
    return __read(XMLBIFReader, file, vtype)
