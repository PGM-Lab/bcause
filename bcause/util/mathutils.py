import math


def rrmse(a,astar,b,bstar):
    return math.sqrt(((a-astar)**2 + (b-bstar)**2)/(2*(bstar-astar)**2))
