from itertools import starmap

import networkx as nx


def fusion_roots(models, on, **builder_kargs):
    is_root = lambda x ,m : len(list(m.graph.predecessors(x))) ==0

    #todo: review this
    if not all(starmap(is_root, zip(on ,models))):
        raise ValueError("Common nodes when fusion must all be root")

    if len(set([type(m) for m in models]))!=1:
        raise ValueError("All models when fusion must all of the same type")

    new_dag = nx.DiGraph()
    for v in models[0].variables: new_dag.add_node(v)
    new_factors = models[0].get_factors(*on)
    for m in models:
        for x ,y in m.graph.edges: new_dag.add_edge(x ,y)
        new_factors.extend(m.get_factors(*set(m.variables).difference(on)))

    return models[0].builder(dag=new_dag, factors=new_factors, **builder_kargs)


def counterfactual_model(model, do):
    do = do if isinstance(do, list) else [do]
    models = [model]
    for i in range(0, len(do)):
        mapping = {v: f"{v}_{i + 1}" for v in model.endogenous}
        models.append(model.intervention(**do[i]).rename_vars(mapping))
    new_endogenous = sum([m.endogenous for m in models], [])
    return fusion_roots(models, on=model.exogenous, endogenous=new_endogenous)

