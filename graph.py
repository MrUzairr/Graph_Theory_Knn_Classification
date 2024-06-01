import nltk
import pandas as pd
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer
from nltk.tokenize import word_tokenize

import networkx as nx
import matplotlib.pyplot as plt
from collections import defaultdict
from graphrole import RecursiveFeatureExtractor


class TextGraph:
    def __init__(self, filepath):
        self.df = pd.read_csv(filepath, delimiter=";", encoding="latin1")
        nltk.download("stopwords")
        nltk.download("punkt")
        self.stop_words = set(stopwords.words("english"))
        self.stemmer = PorterStemmer()
        self.training_data = pd.DataFrame()
        self.test_data = pd.DataFrame()
        self.last_graph = None
        self.last_label = None
        self.training_graphs = []

    def preprocess_text(self, text):
        tokens = word_tokenize(text)
        tokens = [token.lower() for token in tokens if token.isalpha()]
        tokens = [
            token for token in tokens if token not in self.stop_words
        ]
        tokens = [self.stemmer.stem(token) for token in tokens]
        return tokens

    def construct_graph(self, tokens):
        G = nx.DiGraph()
        for i in range(len(tokens) - 1):
            if not G.has_edge(tokens[i], tokens[i + 1]):
                G.add_edge(tokens[i], tokens[i + 1], weight=1, arrows="to")
            else:
                G.edges[tokens[i], tokens[i + 1]]["weight"] += 1
                G.edges[tokens[i], tokens[i + 1]]["arrows"] = "to"
        return G

    def divide_data(self):
        for topic, group in self.df.groupby("label"):
            self.training_data = pd.concat(
                [self.training_data, group.head(12)], ignore_index=True
            )
            self.test_data = pd.concat(
                [self.test_data, group.tail(3)], ignore_index=True
            )

    def process_training_data(self):
        for index, row in self.training_data.iterrows():
            label = row["label"]
            text = row["text"]  
            if pd.isnull(text):
                continue
            tokenized_text = self.preprocess_text(text)
            graph = self.construct_graph(tokenized_text)
            self.last_graph = graph
            self.last_label = label
            # print("label:", label)
            # print("Graph Nodes:", len(graph.nodes()))
            # print("Graph Edges:", len(graph.edges()))
            # print("\n")
            self.training_graphs.append(graph)

    # def visualize_last_graph(self):
    #     if self.last_graph is not None:
    #         # Ensure the graph is directed
    #         if not self.last_graph.is_directed():
    #             self.last_graph = self.last_graph.to_directed()
    #         net = Network(notebook=True)
    #         net.from_nx(self.last_graph)
    #         net.show("graph.html")

    def visualize_last_graph(self):
        if self.last_graph is not None:
            # Ensure the graph is directed
            if not self.last_graph.is_directed():
                self.last_graph = self.last_graph.to_directed()
            pos = nx.spring_layout(self.last_graph)
            nx.draw(self.last_graph, pos, with_labels=True, arrows=True)
            plt.title("Last Graph")
            plt.show()

    def extract_role_features(self):
        role_features = []
        for graph in self.training_graphs:
            feature_extractor = RecursiveFeatureExtractor(graph)
            features = feature_extractor.extract_features()
            role_features.append(features)
            break
        return role_features

    def find_common_subgraphs(self):
        common_subgraphs = defaultdict(int)
        # print("Training graphs:")
        for i, graph in enumerate(self.training_graphs):
            # print(f"Graph {i}:")
            for component in nx.weakly_connected_components(graph):
                common_subgraphs[frozenset(component)] += 1
                # print(
                #     f"  Component {common_subgraphs[frozenset(component)]}: {component}"
                # )
        common_subgraphs = {
            k: v for k, v in common_subgraphs.items() if v == len(self.training_graphs)
        }
        # print(f"Number of common subgraphs: {len(common_subgraphs)}")
        return common_subgraphs


# Usage

