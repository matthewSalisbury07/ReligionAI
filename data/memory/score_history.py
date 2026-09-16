import json
import os
import datetime

FILE = "memory/score_history.json"

def save_scores(scores):
    entry = {
        "time": datetime.datetime.now().isoformat(),
        "scores": scores
    }

    if not os.path.exists(FILE):
        data = []
    else:
        with open(FILE, "r") as f:
            data = json.load(f)

    data.append(entry)

    os.makedirs("memory", exist_ok=True)
    with open(FILE, "w") as f:
        json.dump(data, f, indent=2)


def load_scores():
    if not os.path.exists(FILE):
        return []
    return json.load(open(FILE))