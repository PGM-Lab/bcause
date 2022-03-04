
from pgmpy.readwrite import BIFWriter, XMLBIFWriter

from conversion.pgmpy import toPgmpyBNet
from models import BayesianNetwork


def toBIF(model:BayesianNetwork, path):
    return BIFWriter(toPgmpyBNet(model)).write_bif(path)

def toXMLBIF(model:BayesianNetwork, path):
    return XMLBIFWriter(toPgmpyBNet(model)).write_xmlbif(path)