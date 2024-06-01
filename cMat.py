import re
import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix


# Preprocessing function
def preprocess(text):
    if isinstance(text, str):
        tokens = re.findall(r"\b\w+\b", text.lower())
        return " ".join(tokens)
    else:
        return "doctor"

# Update makeGraph function to use preprocessed text
def makeGraph(string):
    chunks = preprocess(string).split()
    G = nx.DiGraph()
    for chunk in set(chunks):
        G.add_node(chunk)
    for i in range(len(chunks) - 1):
        G.add_edge(chunks[i], chunks[i + 1])
    return G


# Plot confusion matrix
def plot_confusion_matrix(true_labels, predicted_labels, classes):
    cm = confusion_matrix(true_labels, predicted_labels, labels=classes)
    plt.figure(figsize=(8, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        xticklabels=classes,
        yticklabels=classes,
    )
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion Matrix")
    plt.show()
