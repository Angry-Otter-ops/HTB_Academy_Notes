import os
import requests
import random
import numpy as np

random.seed(1337); np.random.seed(1337)
host = os.getenv("BASE_URL")

ch = requests.get(f"{host}/challenge", timeout=10).json()
base, budget = ch["base_message"], int(ch["max_added_words"])

def predict(t):
    return requests.post(f"{host}/predict", json={"text": t}, timeout=15).json()

base_p = predict(base)["spam_probability"]
print(f"Base spam probability: {base_p}")
