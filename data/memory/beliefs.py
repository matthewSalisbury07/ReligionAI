import json
import os

FILE = "memory/beliefs.json"

def load_beliefs():
    if not os.path.exists(FILE):
        return {}
    return json.load(open(FILE))


def update_beliefs_from_scores(scores):
    beliefs = load_beliefs()

    for key, value in scores.items():
        if key not in beliefs:
            beliefs[key] = 0

        # Smooth update (prevents spikes)
        beliefs[key] = int((beliefs[key] * 0.8) + (value * 0.2))

    os.makedirs("memory", exist_ok=True)
    json.dump(beliefs, open(FILE, "w"), indent=2)

    return beliefs