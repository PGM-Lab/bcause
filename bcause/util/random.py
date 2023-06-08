import random

import numpy as np


def seed(n):
    np.random.seed(n)
    random.seed(n)