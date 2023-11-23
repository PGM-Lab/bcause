"""
Creates an adjustable Markovian model (n_levels), sample from it,  estimate 
its parameters, and plot some visualizations

TODO:
- Generalize canonical factors for markovian to quasi-markovian
"""

import networkx as nx
from collections import defaultdict
import matplotlib.pyplot as plt
import numpy as np

import bcause as bc
from bcause.factors import MultinomialFactor, DeterministicFactor 
from bcause.inference.causal.multi import GDCC, EMCC
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
    Iterates over estimators, sample_sizes and num_runs
    and computes the log-likelihood ratio of the estimated model for
    data sampled from (true) model
    """
    def __init__(self, model, estimators, sample_sizes, num_runs, max_iter):
        self.model = model
        self.sample_sizes = sample_sizes
        self.num_runs = num_runs
        self.estimators = estimators
        self.max_iter = max_iter

    def launch(self):
        rllkh = defaultdict(lambda : {})
        for sample_size in self.sample_sizes:
            print(f'{sample_size=}')
            rllkh[sample_size] = defaultdict(lambda : {})
            data = self.model.sample(sample_size, as_pandas=True)[self.model.endogenous]
            lmax = self.model.max_log_likelihood(data)
            for name, estimator in self.estimators.items():
                infer = estimator(self.model, data, num_runs = self.num_runs, max_iter = self.max_iter) # estimate from the same data multiple times with different initial point
                infer.compile()
                for i_run, m in enumerate(infer.models):
                    #rllkh[sample_size][i_run][name] = m.max_log_likelihood(data)/m.log_likelihood(data)
                    rllkh[sample_size][i_run][name] = lmax/m.log_likelihood(data) # TODO: this or the one above?
        return rllkh 


    def visualize(self, rllkh, save = None):
        """
        Visualize the log-likelihood ratio using scatter plot.
        """
        plt.figure(figsize=(10, 6))

        for sample_size in rllkh:
            x = [sample_size] * self.num_runs
            for name, estimator in self.estimators.items():
                y = [rllkh[sample_size][i_run][name] for i_run in range(self.num_runs)]
                plt.scatter(x, y, label=f'{name}', color = 'r' if name == 'EM' else 'b')

        plt.title(f'Log-Likelihood Ratio by Sample Size and {self.num_runs} Replications')
        plt.xlabel('Sample Size')
        plt.ylabel('Log-Likelihood Ratio')
        plt.legend(list(self.estimators.keys()))
        self.show_or_save(save)


    def show_or_save(self, save):
        if save:
            save_dir = 'results'
            fig_path = f'{save_dir}//{save}'
            plt.savefig(fig_path, dpi = 400)
            print(f'{fig_path} saved.')
            plt.close()
        else:
            plt.show(block = True)


    def visualize_max_iter(self, rllkh, save=None):
        """
        Visualize the log-likelihood ratio for each estimator as a function of max_iter.
        Only one sample size is considered in the experiments.
        """
        plt.figure(figsize=(10, 6))

        # iterate over each max_iter value and corresponding results
        for max_iter, iter_data in rllkh.items():
            for name, estimator in self.estimators.items():
                y = [iter_data[self.sample_sizes[0]][i_run][name] for i_run in range(self.num_runs)]
                plt.scatter([max_iter] * self.num_runs, y,  label=f'{name}', color = 'r' if name == 'EM' else 'b', alpha=0.7)

        plt.title('Log-Likelihood Ratio by Max Iteration and Estimator')
        plt.xlabel('Max Iteration')
        plt.ylabel('Log-Likelihood Ratio')
        plt.legend(list(self.estimators.keys()))
        self.show_or_save(save)


    def visualize_max_iter_boxplot(self, rllkh, save=None):
        """
        Visualize the log-likelihood ratio for each estimator as a function of max_iter using boxplots.
        Only one sample size is considered in the experiments. Estimators are distinguished by color.
        """
        plt.figure(figsize=(10, 6))

        # Prepare data structure for boxplot
        data_to_plot = {estimator: [] for estimator in self.estimators}
        max_iters = sorted(rllkh.keys())
        colors = ['red', 'blue']  # colors for different estimators

        for max_iter in max_iters:
            iter_data = rllkh[max_iter]
            for name in self.estimators:
                y = [iter_data[self.sample_sizes[0]][i_run][name] for i_run in range(self.num_runs)]
                data_to_plot[name].append(y)

        # Plot boxplots for each estimator
        for idx, (estimator, data) in enumerate(data_to_plot.items()):
            positions = np.arange(len(max_iters)) * len(self.estimators) + idx
            plt.boxplot(data, positions=positions, widths=0.6, patch_artist=True, boxprops=dict(facecolor=colors[idx]))

        # Add legend and labels
        plt.legend(data_to_plot.keys())
        plt.xticks(np.arange(len(max_iters)) * len(self.estimators) + 0.5, max_iters)
        plt.title('Log-Likelihood Ratio by Max Iteration and Estimator')
        plt.xlabel('Max Iteration')
        plt.ylabel('Log-Likelihood Ratio')
        self.show_or_save(save)


def run_exp_rllkh():
    for n_levels in [2, 3]:
        print(f'{n_levels=}')
        mscm = MarkovianSCM(n_levels = n_levels)
        m = mscm.create()
        estimators = {'EM': EMCC, 'GL': GDCC}
        exp = Expertiment(m, estimators, np.array([10, 20, 50, 100]) * 10, 5, 20)
        rllkh = exp.launch()
        print(rllkh)
        exp.visualize(rllkh, save = f'exp_{n_levels}')

def run_exp_max_iter():
    n_levels = 2
    mscm = MarkovianSCM(n_levels = n_levels)
    m = mscm.create()
    estimators = {'EM': EMCC, 'GL': GDCC}
    rllkh = {}
    for max_iter in range(2, 8, 1):
        print(f'{max_iter=}')
        exp = Expertiment(m, estimators, [500], 20, max_iter)
        rllkh[max_iter] = exp.launch()
        print(rllkh[max_iter])
    #exp.visualize_max_iter(rllkh) # save = f'exp_{n_levels}')
    exp.visualize_max_iter_boxplot(rllkh, save = f'exp_max_iter_{n_levels}')


if __name__ == "__main__":
    bc.randomUtil.seed(100)
    run_exp_rllkh()
    run_exp_max_iter()