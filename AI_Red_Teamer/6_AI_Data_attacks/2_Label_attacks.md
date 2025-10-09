# LAbel Attacks

## Label Flipping

Label Flipping is arguably the simplest form of a data poisoning attack. It directly targets the ground truth information used during model training.

The most common goal of an attacker executing a Label Flipping attack is to degrade model performance. By introducing incorrect labels, the attack forces the model to learn incorrect associations between features and classes, resulting in a "confused" model that has a general decrease in performance (eg accuracy, precision, recall, etc).

This type of attack often targets data after it has been collected, focusing on compromising the integrity of datasets held within the Storage stage of the pipeline. 

## Baseline Logistic Regression Model
using the Python code samples provided, generate the baseline data to begin poisoning later. 

## The Label Flipping Attack
we will create a function that will take the original training labels (y_train, representing the true sentiments) and a poisoning percentage as input. It will randomly select the specified fraction of training data points (reviews) and flip their labels - changing Negative (0) to Positive (1) and Positive (1) to Negative (0).

The implication of this is significant. As we have established, the model learns its parameters (𝐰,b) by minimizing the average log-loss, L, across the training dataset, the whole point of training is to find the 𝐰 and b that make this loss L as small as possible, meaning the predicted probabilities pi align well with the true labels yi.

To execute this attack, we will implement a function to contain all logic: flip_labels. This function takes the original training labels (y_train) and a poison_percentage as input, specifying the fraction of labels to flip.

First, we define the function signature and ensure the provided poison_percentage is a valid value between 0 and 1. This prevents nonsensical inputs. We also calculate the absolute number of labels to flip (n_to_flip) based on the total number of samples and the specified percentage.

Next, we select which specific reviews (data points) will have their sentiment labels flipped. We use a NumPy random number generator (rng_instance) seeded with our global SEED (or the function's seed parameter) for reproducible random selection. The choice method selects n_to_flip unique indices from the range 0 to n_samples - 1 without replacement. These flipped_indices identify the exact reviews targeted by the attack.

Now, we perform the actual label flipping. We create a copy of the original label array (y_poisoned = y.copy()) to avoid altering the original data. For the elements at the flipped_indices, we invert their labels: 0 becomes 1, and 1 becomes 0. A concise way to do this is 1 - label for binary 0/1 labels, or using np.where for clarity.

Lastly, the function returns the y_poisoned array containing the corrupted labels and the flipped_indices array, allowing us to track which reviews were affected.

# Evaluating Label Flipping Attack

Let's begin by poisoning a small fraction, say 10%, of the training labels and observe the impact on our sentiment analysis model.

The process involves several steps:

    Use the flip_labels function on the original y_train data to create a poisoned version (y_train_poisoned_10) where 10% of the sentiment labels are flipped.
    Visualize the resulting corrupted training set using plot_poisoned_data to see which points were flipped.
    Train a new Logistic Regression model (model_10_percent) using the original features X_train but the poisoned labels y_train_poisoned_10.
    Evaluate this poisoned model's accuracy on the original, clean test set (X_test, y_test). This is crucial - we want to see how the poisoning affects performance on legitimate, unseen data.
    Visualize the decision boundary learned by the model_10_percent using plot_decision_boundary.
