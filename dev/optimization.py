import logging
import sys
import networkx as nx
import matplotlib.pyplot as plt

import bcause.util.domainutils as dutils
from bcause.factors import DeterministicFactor
from bcause.factors.mulitnomial import MultinomialFactor
from bcause.models.cmodel import StructuralCausalModel
from bcause.util.plotutils import plot_3d, get_linear_colors

log_format = '%(asctime)s|%(levelname)s|%(filename)s: %(message)s'
logging.basicConfig(level=logging.INFO, stream=sys.stdout, format=log_format, datefmt='%Y%m%d_%H%M%S')


## Example model
dag = nx.DiGraph([("X", "Y"), ("U", "Y"), ("V", "X")])
#nx.draw(dag, with_labels=True, font_weight='bold')
#plt.show() 

domains = dict(X=["x1", "x2"], Y=["y1","y2"], U=["u1", "u2", "u3", "u4"], V=["v1", "v2"])
domx = dutils.var_parents_domain(domains,dag,"X")
fx = DeterministicFactor(domx, right_vars=["V"], values=["x1", "x2"]).to_multinomial()
domy = dutils.var_parents_domain(domains,dag,"Y")
values = ["y1", "y1", "y2", "y1", "y2", "y2", "y1", "y1"] # ??
fy = DeterministicFactor(domy, left_vars=["Y"], values=values,vtype="list").to_multinomial()
domv = dutils.subdomain(domains, "V")
pv = MultinomialFactor(domv, values=[.5, .5])
domu = dutils.subdomain(domains, "U")
pu = MultinomialFactor(domu, values=[.2, .2, .6, .0])
model = StructuralCausalModel(dag, [fx, fy, pu, pv], cast_multinomial=True)


## Example of empirical distribution
data = [.4,.6,.6,.4]
#data = [.4,.6,.58,.42]
#data = [.1,.9,.58,.42]
#data = [.1,.9,.8,.2]
py_x = MultinomialFactor(dutils.subdomain(domains, "X","Y"), right_vars=["X"], values=data) 


if __name__ == "__main__":
    import numpy as np

    # Fake optimizer
    def fake_optimizer(py_x, fy, num_runs):
        def sample_dist():
            v = np.random.uniform(0, 0.4)
            return MultinomialFactor(domu, values=[v, 0.4 - v, .6, .0])

        return [sample_dist() for _ in range(num_runs)]

    #Run the fake optimizer and plot the solutions
    solutions_fake = fake_optimizer(py_x, fy, 10)
    plot_3d([p.values[:3] for p in solutions_fake])