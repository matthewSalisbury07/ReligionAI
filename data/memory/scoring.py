import re

CATEGORIES = {
    "fear": ["fear", "afraid", "terror"],
    "morality": ["right", "wrong", "good", "evil"],
    "truth": ["truth", "knowledge"],
    "obedience": ["obey", "command"],
    "mercy": ["mercy", "forgive"],
    "certainty": ["certain", "absolute"]
}

NEGATIONS = ["not", "no", "never"]

def score_text(text):
    text = text.lower()
    words = re.findall(r"\w+", text)

    scores = {}

    for category, keywords in CATEGORIES.items():
        score = 0

        for i, word in enumerate(words):
            if word in keywords:
                window = words[max(0, i-3):i]
                if any(n in window for n in NEGATIONS):
                    score -= 1
                else:
                    score += 1

        score = max(-5, min(5, score))
        percent = int((score / 5) * 100)

        scores[category] = percent

    return scores