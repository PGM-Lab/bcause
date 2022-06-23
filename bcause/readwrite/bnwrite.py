import logging
import os

from pgmpy.readwrite import BIFWriter, XMLBIFWriter

from bcause.conversion.pgmpy import toPgmpyBNet
from bcause.models import BayesianNetwork
from bcause.util.assertions import assert_file_exists


def __write(writer, model, path):
    folder = os.path.dirname(path)
    assert_file_exists(folder)
    format = writer.__name__.replace('Writer', '')
    logging.info(f"Saving model in {format} to {os.path.abspath(path)}")
    getattr(writer(toPgmpyBNet(model)), f"write_{format.lower()}")(path)

def toBIF(model:BayesianNetwork, path):
    __write(BIFWriter, model, path)

def toXMLBIF(model:BayesianNetwork, path):
    __write(XMLBIFWriter, model, path)
