"""
Creates an adjustable Markovian model (n_levels), sample from it, and estimate 
its parameters

TODO:
- change n_levels
"""

import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import bcause as bc
from bcause.factors import MultinomialFactor, DeterministicFactor 
from bcause.learning.parameter.gradient import GradientLikelihood
from bcause.learning.parameter.expectation_maximization import ExpectationMaximization
from bcause.models.cmodel import StructuralCausalModel
from bcause.factors.deterministic import canonical_specification
import bcause.util.domainutils as dutils
import bcause.util.graphutils as gutils

class MarkovianSCM:
    """
    A class of Markovian SCMs 
    """
    def __init__(self, n_levels: int, n_endo_states: list[int] = None) -> None:
        self.n_levels = n_levels
        self.n_endo_states = n_endo_states
        if self.n_endo_states == None:
            self.n_endo_states = [2] * n_levels # binary if not specified

    def create_DAG(self):
        # creates a Markovian graph with n_levels of exo and endo nodes
        edges, domains= [], {}
        for level in range(self.n_levels):
            exo = f'U{level}'
            endo = f'X{level}'
            edges.append((exo, endo))
            domains[endo] = [f'x{level}{i}' for i in range(self.n_endo_states[level])]
            if level == 0: 
                # NOTE: X0 doesn't have any endogenous parent
                domains[exo] = [f'u{level}{i}' for i in range(self.n_endo_states[level])]
            else:
                endo_parent = f'X{level-1}'
                edges.append((endo_parent, endo))
                domains[exo] = [f'u{level}{i}' for i in range(self.n_endo_states[level] ** self.n_endo_states[level])]

        dag = nx.DiGraph(edges)
        return dag, edges, domains


    def create_factors(self, domains, dag):
        # defines the structural equations (f) for the endogenous variables
        # and PMF (p) for exogenous variable
        p, f = {}, {}
        for level in range(self.n_levels):
            exo = f'U{level}'
            endo = f'X{level}'
            dom_endo = dutils.subdomain(domains, *gutils.relevat_vars(dag, endo))
            dom_exo = dutils.subdomain(domains, exo)
            if level == 0:
                p[exo] = MultinomialFactor(dom_exo, values=[.3, .7])
                f[endo] = DeterministicFactor(dom_endo, left_vars=[endo], values=domains[endo])
            else:
                p[exo] = MultinomialFactor(dom_exo, values=[.2, .2, .1, .5])
                endo_parent = f'X{level-1}'
                can_values = canonical_specification(V_domain = domains[endo], Y_domains = [domains[endo_parent]])
                f[endo] = DeterministicFactor(dom_endo, left_vars=[endo], values=can_values)
        return p, f

    def create(self):
        # creates a prespecified SCM
        dag, _, domains = self.create_DAG()
        p,f = self.create_factors(domains, dag)
        m = StructuralCausalModel(dag, list(f.values()) + list(p.values()), cast_multinomial=True)
        return m


class Expertiment:
    """
    Iterates over estimators, sample_sizes and n_replications
    and computes the log-likelihood of the estimated model for
    data sampled from (true) model
    """
    def __init__(self, model, estimators, sample_sizes, n_replications) -> None:
        self.model = model
        self.sample_sizes = sample_sizes
        self.n_replications = n_replications
        self.estimators = estimators

    def launch(self):
        llkh = defaultdict(lambda : {})
        self.data =  {}
        for sample_size in self.sample_sizes:
            print(f'{sample_size=}')
            llkh[sample_size] = defaultdict(lambda : {})
            self.data[sample_size] = self.model.sample(sample_size, as_pandas=True)[self.model.endogenous]
            for i_rep in range(self.n_replications):
                #self.data[sample_size] = self.model.sample(sample_size, as_pandas=True)[self.model.endogenous]
                for name, estimator in self.estimators.items():
                    estimator.run(self.data[sample_size]) # estimate from the same data multiple times with different initial point

                    llkh[sample_size][i_rep][name] = estimator.model.log_likelihood(self.data[sample_size])
                    #ratio_llk = estimator.model.max_log_likelihood(self.data[sample_size])/llkh[sample_size][i_rep][name]
                    #print(f"{sample_size},{i_rep},{name}: {ratio_llk}")

                    #print(f'{name=}: {estimator.model.log_likelihood(self.data[sample_size])}')
        return llkh 


    def visualize(self, llkh, save = None):
        """
        Visualize the average log-likelihood values per sample using scatter plot.
        """
        plt.figure(figsize=(10, 6))

        label_max = 'max'
        plt.plot([sample_size for sample_size in llkh], 
                 [self.model.max_log_likelihood(self.data[sample_size]) / sample_size for sample_size in llkh],
                 label = label_max, color = 'k')

        for sample_size in llkh:
            x = [sample_size] * self.n_replications
            for name, estimator in self.estimators.items():
                y = [llkh[sample_size][i_rep][name] / sample_size for i_rep in range(self.n_replications)]
                plt.scatter(x, y, label=f'{name}', color = 'r' if name == 'EM' else 'b')

        plt.title(f'Average Log-Likelihood per Sample by Sample Size and {self.n_replications} Replications')
        plt.xlabel('Sample Size')
        plt.ylabel('Average Log-Likelihood per Sample')
        plt.legend([label_max] + list(self.estimators.keys()))
        #plt.grid(True)
        if save:
            save_dir = 'results'
            fig_path = f'{save_dir}//{save}'
            plt.savefig(fig_path, dpi = 400)
            print(f'{fig_path} saved.')
            plt.close()
        else:
            plt.show()


if __name__ == "__main__":
    bc.randomUtil.seed(100)
    for n_levels in [1, 2, 3]:
        print(f'{n_levels=}')
        mscm = MarkovianSCM(n_levels = n_levels)
        m = mscm.create()
        estimators = {'EM': ExpectationMaximization(m.randomize_factors(m.exogenous, allow_zero=False)), 
                      'GL': GradientLikelihood(m.randomize_factors(m.exogenous, allow_zero=False))}
        exp = Expertiment(m, estimators, np.array([10, 20, 50, 100]) * 10, 5)
        llkh = exp.launch()
        print(llkh)
        exp.visualize(llkh, save = f'exp_{n_levels}')

