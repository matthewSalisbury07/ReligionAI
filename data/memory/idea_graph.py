import json
import os

FILE = "memory/idea_graph.json"

def load_graph():
    if not os.path.exists(FILE):
        return {}
    return json.load(open(FILE))


def save_graph(graph):
    os.makedirs("memory", exist_ok=True)
    json.dump(graph, open(FILE, "w"), indent=2)


def update_graph(ideas):
    graph = load_graph()

    for idea in ideas:
        if idea not in graph:
            graph[idea] = {}

        for other in ideas:
            if other == idea:
                continue

            graph[idea][other] = graph[idea].get(other, 0) + 1

    save_graph(graph)
    return graph