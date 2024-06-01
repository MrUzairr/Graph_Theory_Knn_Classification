import networkx as nx


class GraphKNN:
    def __init__(self, k: int):
        self.k = k
        self.train_graphs = []
        self.train_labels = []

    def fit(self, train_graphs, train_labels):
        self.train_graphs = train_graphs
        self.train_labels = train_labels

    def predict(self, graph):
        distances = []
        for train_graph in self.train_graphs:
            distance = graphDistance(graph, train_graph)
            distances.append(distance)
        nearest_indices = sorted(range(len(distances)), key=lambda i: distances[i])[
            : self.k
        ]
        
        nearest_labels = [self.train_labels[i] for i in nearest_indices]
        prediction = max(set(nearest_labels), key=nearest_labels.count)
        return prediction

# Calculate graph distance
def graphDistance(graph1, graph2):
    edges1 = set(graph1.edges())
    edges2 = set(graph2.edges())
    common = edges1.intersection(edges2)
    mcs_graph = nx.Graph(list(common))
    return -len(mcs_graph.edges())

