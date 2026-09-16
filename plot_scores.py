import matplotlib.pyplot as plt
from memory.score_history import load_scores

data = load_scores()

if not data:
    print("No score history yet.")
    exit()

categories = list(data[0]["scores"].keys())

for cat in categories:
    values = [entry["scores"][cat] for entry in data]
    plt.plot(values, label=cat)

plt.title("Score Trends Over Time")
plt.xlabel("Journal Entry")
plt.ylabel("Score (%)")
plt.legend()
plt.show()