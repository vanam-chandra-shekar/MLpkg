from graphviz import Digraph
from .engine import Value


def _trace(root : Value):
    nodes , edges = set() , set()

    def build(v):
        if v not in nodes:
            nodes.add(v)
            for child in v._children:
                edges.add((child , v))
                build(child)
    build(root)
    return nodes , edges

def visualize_computational_graph(root : Value):

    nodes , _ = _trace(root)

    dot = Digraph(comment="Computational Graph" , graph_attr={"rankdir" : "LR"})

    for n in nodes:
        uid = str(id(n))
        data_label = f"{n.label} | data={n.data} | grad = {n.grad}"
        dot.node(name=uid , label=data_label , shape="box")

        if n._op:
            op_id = uid + n._op

            dot.node(name=op_id , label = n._op)
            dot.edge(op_id , uid)

            for c in n._children:
                dot.edge(str(id(c)) , op_id)

    return dot