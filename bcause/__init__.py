__version__ = '0.0.2'

import logging
logging.basicConfig(level=logging.WARNING) # JAN2RAFA: We need something like this to suppress all those logs printed to sys.stdout

from .conversion import *
from .factors import *
from .inference import *
from .learning import *
from .models import *
from .readwrite import *
from .util import *



