import networkx as nx
import matplotlib.pyplot as plt

# Create an empty graph
G = nx.Graph()

# Add nodes
# G.add_nodes_from([1, 2, 3, 4, 5])
G.add_node((1,2))
G.add_node((3,4))
G.add_node((3,4))
G.add_node((5,6))
G.add_node((7,8))

# Add edges
# G.add_edges_from([(3, 1), (3, 2), (3, 4), (3, 5)])
G.add_edge((1,2), (3,4))
G.add_edge((3,4),(1,2))
G.add_edge((5,6), (7,8))
print(G)
# Draw the graph
nx.draw(G, with_labels=True)
plt.show()

draw_path = nx.approximation.traveling_salesman_problem(G, cycle=False)
print(draw_path)