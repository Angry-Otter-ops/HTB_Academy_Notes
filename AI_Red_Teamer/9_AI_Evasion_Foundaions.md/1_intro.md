# Intro
Evasion attacks manipulate inference-time inputs to cause a trained model to produce incorrect outputs. Adversarial machine learning, the discipline that studies these hostile interactions with machine learning systems, treats this interference during the inference phase as a distinct threat because it bypasses safeguards built into the training pipeline


- **Data poisoning** injects or alters training samples so the model internalizes biased patterns. 
- **Label manipulation** tampers with annotations so ground truth no longer matches reality. 
- **Trojan attacks** implant hidden triggers that activate specific behavior when an attacker’s pattern appears in the input.


# The GoodWords Attack
GoodWords attack manipulates probabilistic spam filters by inserting carefully selected benign tokens. Because Naive Bayes models assume conditional independence between features, adding those tokens shifts posterior probabilities enough to force a misclassification without raising obvious suspicion.

## Good Word Selection Strategy

![alt text](image.png)


![alt text](image-1.png)
# Black-Box GoodWords