from pgmpy.readwrite import BIFReader, XMLBIFReader

from conversion.pgmpy import toBCauseBNet

def fromBIF(file):
    return toBCauseBNet(BIFReader(file).get_model())
def fromXMLBIF(file):
    return toBCauseBNet(XMLBIFReader(file).get_model())
