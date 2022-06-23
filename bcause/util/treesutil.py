from abc import ABC
from functools import reduce

import numpy as np

'''

class TreeNode(ABC):
    def __init__(self, var, children=None):
        self.var = var
        self.children = children

    def __repr__(self):

        str_ch = ", ".join([f"{self.var}={k}:{v}" for k,v in self.children.items()])
        return f"<{str_ch}>"

'''

def treeNode(var, children):
    return dict(var=var, children=children)

def build_default_tree(dom, data):
    variables = list(dom.keys())
    if np.ndim(data)>1: data = np.ravel(data)

    var = variables[0]
    if len(variables)==1:
        return treeNode(var = var,  children = dict(zip(np.ravel(list(dom.values())), np.ravel(data))))

    inner_dom =  {v:s for v,s in dom.items() if v != var}
    inner_size = reduce(lambda x,y: x*y, [len(s) for s in inner_dom.values()])

    children = dict()
    for i in range(len(dom[var])):

        inner_data = data[i*inner_size: (i+1)*inner_size]
        children[dom[var][i]] = build_default_tree(inner_dom, inner_data)

    return treeNode(var, children)



