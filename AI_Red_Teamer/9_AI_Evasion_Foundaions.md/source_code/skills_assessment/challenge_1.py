#!/usr/bin/env python3
import os
import requests
import pickle
import numpy as np
from typing import List, Dict, Tuple

SEED = 1337
np.random.seed(SEED)

BASE_URL = os.environ.get("BASE_URL")

# Create WhiteBoxAttacker class and set init states
class WhiteBoxAttacker:
    def __init__(self, base_url: str):
        self.base_url = base_url
        self.model = None
        self.vectorizer = None
        self.feature_names = None
        self.classes = None

# Create function to download and load the Naive Bayes classifier
    def download_model(self):
        print("[*] Downloading model...")
        r = requests.get(f"{self.base_url}/model/download")
        r.raise_for_status()

        with open("/tmp/model.pkl", "wb") as f:
            f.write(r.content)

        with open("/tmp/model.pkl", "rb") as f:
            bundle = pickle.load(f)

        self.model = bundle["classifier"]
        self.vectorizer = bundle["vectorizer"]
        self.feature_names = bundle["feature_names"]
        self.classes = bundle["classes"]
        print(f"[+] Model loaded: {len(self.feature_names)} features")

# Calculate word scores based on the Naive Bayes feature probabilities.
    def calculate_word_scores(self, target_class: str) -> List[Tuple[str, float]]:
        target_idx = self.classes.index(target_class)
        other_idx = 1 - target_idx

        scores = []
        for i, feature in enumerate(self.feature_names):
            if " " in feature:
                continue
            target_prob = np.exp(self.model.feature_log_prob_[target_idx][i])
            other_prob = np.exp(self.model.feature_log_prob_[other_idx][i])
            score = target_prob / (other_prob + 1e-10)
            scores.append((feature, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

# Implement the attack method for individual reviews by incrementally adding the highest-scoring words until the sentiment flips

    def attack_review(self, text: str, target: str, max_words: int) -> Tuple[str, int]:
        word_scores = self.calculate_word_scores(target)

        augmented = text
        for num_words in range(1, max_words + 1):
            words_to_add = [w for w, _ in word_scores[:num_words]]
            augmented = text + " " + " ".join(words_to_add)

            vec = self.vectorizer.transform([augmented])
            prediction = self.model.predict(vec)[0]

            if prediction == target:
                return augmented, num_words

        return augmented, max_words
    
#  main white-box solver method that retrieves the challenge, attacks each review, and submits the solutions
    def solve_whitebox(self) -> Dict:
        print("\n[*] Starting white-box phase...")

        r = requests.get(f"{self.base_url}/challenge/whitebox")
        r.raise_for_status()
        challenge = r.json()

        reviews = challenge["reviews"]
        max_words = challenge["max_added_words"]

        self.download_model()

        solutions = []
        for review in reviews:
            print(f"  Attacking review {review['id']}...", end=" ")
            augmented, words_used = self.attack_review(
                review["text"], review["target_sentiment"], max_words
            )
            solutions.append({"id": review["id"], "augmented_text": augmented})
            print(f"Done ({words_used} words)")

        r = requests.post(
            f"{self.base_url}/submit/whitebox", json={"solutions": solutions}
        )
        r.raise_for_status()
        result = r.json()

        if "results" in result:
            successes = sum(1 for r in result["results"] if r.get("success", False))
            print(f"[+] White-box phase: {successes}/10 completed")

        return result

# Define main to execute attack
def main():
    wb_attacker = WhiteBoxAttacker(BASE_URL)
    wb_result = wb_attacker.solve_whitebox()

    if not wb_result.get("phase_complete", False):
        print("[-] Failed to complete white-box phase")
        return

    print("[+] White-box phase completed successfully!")

if __name__ == "__main__":
    main()
