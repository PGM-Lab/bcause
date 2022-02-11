import networkx as nx


def dag2str(dag: nx.DiGraph) -> str:
    str_dag = ""
    for v in dag.nodes:
        parents = list(dag.predecessors(v))
        str_dag += f"[{v}"
        if len(parents) > 0:
            str_dag += "|"
            str_dag += ":".join(parents)
        str_dag += "]"
    return str_dag


def str2dag(str_dag: str) -> nx.DiGraph:
    arcs = []
    s = str_dag[1:-1].split("][")[1]
    for s in str_dag[1:-1].split("]["):
        if "|" in s:
            x, right_vars = s.split("|")
            for y in right_vars.split(":"):
                arcs.append((y, x))

    return nx.DiGraph(arcs)
