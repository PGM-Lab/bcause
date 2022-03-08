import logging
import os.path

import networkx as nx


def assert_dag_with_nodes(dag, nodes):
    missing_nodes = set(nodes).difference(dag.nodes)
    if len(missing_nodes) > 0:
        msg = f"{missing_nodes} not found in the graph"
        logging.critical(msg)
        raise nx.NodeNotFound(msg)

def assert_file_exists(path):
    if not os.path.exists(path):
        msg = f"No such file or directory: {path}"
        logging.critical(msg)
        raise FileNotFoundError(msg)

