import networkx as nx
import matplotlib.pyplot as plt
import numpy as np

# Create an empty graph
# G = nx.Graph()
G = nx.DiGraph()

# Add nodes
# G.add_nodes_from([1, 2, 3, 4, 5])
# G.add_node((1,2))
# G.add_node((3,4))
# G.add_node((3,4))
# G.add_node((5,6))
# G.add_node((7,8))
# G.add_node((9,10))

# Add edges
# G.add_edges_from([(3, 1), (3, 2), (3, 4), (3, 5)])
G.add_edge((1,2), (3,4), weight=1)
G.add_edge((3,4), (1,2), weight=2)
# G.add_edge((3,4),(1,2))
G.add_edge((3,4), (5,6), weight=3)
G.add_edge((5,6), (3,4), weight=1)
G.add_edge((5,6), (7,8), weight=1)
G.add_edge((7,8), (5,6), weight=2)
G.add_edge((7,8),(9,10), weight=1)
G.add_edge((9,10),(7,8), weight=1)
print(G)

for edge in G.edges:
    print(edge)
    print(edge[0],edge[1])
    print(np.array(edge[0])-np.array(edge[1]))

# Draw the graph
nx.draw(G, with_labels=True)
plt.show()

connetion_path = nx.approximation.traveling_salesman_problem(G)
print(connetion_path)
exit()


G_new = nx.contracted_nodes(G, (5,6), (3,4),self_loops=False)
nx.draw(G_new, with_labels=True)
plt.show()
print(G_new)
for edge in G_new.edges:
    print(edge)
    print(G_new.get_edge_data(edge[0],edge[1]))


## check isolation
G.remove_nodes_from(list(nx.isolates(G)))
nx.draw(G, with_labels=True)
plt.show()

## check connectivity
# print("is connected: ", nx.is_connected(G))
# S = [G.subgraph(c).copy() for c in nx.connected_components(G)]
# for subg in S:
#     print(subg)

nodes = {
    "A": dict(color="Red"),
    "B": dict(color="Red"),
    "C": dict(color="Red"),
    "D": dict(color="Red"),
    "E": dict(color="Blue"),
    "F": dict(color="Blue"),
}
edges = [
    ("A", "E", "Strong"),
    ("B", "F", "Strong"),
    ("C", "E", "Weak"),
    ("D", "F", "Weak"),
]
G = nx.Graph()
for node in nodes:
    attributes = nodes[node]
    G.add_node(node, **attributes)

for source, target, type in edges:
    G.add_edge(source, target, type=type)

node_attributes = ('color', )
edge_attributes = ('type', )
summary_graph = nx.snap_aggregation(G, node_attributes=node_attributes, edge_attributes=edge_attributes)
print(summary_graph.nodes)
print(summary_graph.edges)
nx.draw(summary_graph, with_labels=True)
plt.show()

# draw_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
# print(draw_path)