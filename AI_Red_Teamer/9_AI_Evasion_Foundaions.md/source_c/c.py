import json
import pickle
import random
import re
import urllib.request
import zipfile
from pathlib import Path
import numpy as np
import html as html_module
import unicodedata

# Reproducibility
random.seed(1337)
np.random.seed(1337)

import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, accuracy_score

from htb_ai_library import (
    AZURE,
    HACKER_GREY,
    HTB_GREEN,
    MALWARE_RED,
    NODE_BLACK,
    NUGGET_YELLOW,
    WHITE,
    AQUAMARINE,
    load_model,
    save_model,
)

print("\n[*] Loading SMS Spam Dataset...")

data_dir = Path("data")
data_dir.mkdir(exist_ok=True)
dataset_path = data_dir / "sms_spam.csv"

if dataset_path.exists():
    print(f"[+] Using cached dataset: {dataset_path}")
    df = pd.read_csv(dataset_path)
else:
    print("[*] Downloading from UCI repository...")
    url = "https://archive.ics.uci.edu/ml/machine-learning-databases/00228/smsspamcollection.zip"
    zip_path = data_dir / "sms_spam.zip"

    urllib.request.urlretrieve(url, zip_path)
    with zipfile.ZipFile(zip_path, "r") as zf:
        with zf.open("SMSSpamCollection") as f:
            lines = [line.decode("utf-8").strip() for line in f]

    # Parse tab-separated format
    data = []
    for line in lines:
        parts = line.split('\t')
        if len(parts) == 2:
            data.append({"label": parts[0].lower(), "message": parts[1]})

    df = pd.DataFrame(data)
    df.to_csv(dataset_path, index=False)
    zip_path.unlink()
    print(f"[+] Dataset saved to {dataset_path}")

print(f"[+] Loaded {len(df)} messages")
print(f"    Spam: {sum(df['label'] == 'spam')}")
print(f"    Ham: {sum(df['label'] == 'ham')}")

print("\n[*] Sample messages:")
print("\nSPAM samples:")
for msg in df[df['label'] == 'spam']['message'].head(3):
    print(f"  - {msg[:80]}...")
print("\nHAM samples:")
for msg in df[df['label'] == 'ham']['message'].head(3):
    print(f"  - {msg[:80]}...")

def minimal_clean(text):
    """
    Minimal cleaning that preserves spam indicators.

    Parameters: text (str) raw SMS message
    Returns: str cleaned text with entities decoded, unicode normalized, and
             whitespace collapsed while keeping informative symbols.
    """
    # Decode HTML entities (e.g., &amp; -> &)
    text = html_module.unescape(text)

    # Normalize unicode characters
    text = unicodedata.normalize('NFKC', text)

    # Clean up excessive whitespace
    text = re.sub(r'\s+', ' ', text)
    text = re.sub(r'\n+', ' ', text)
    text = re.sub(r'\r+', ' ', text)

    return text.strip()

def clean_text(text):
    """
    Final cleaning for vectorization.

    Converts to lowercase and removes only problematic characters so that
    informative symbols remain available to the vectorizer.

    Parameters:
        text (str): Preprocessed message from `minimal_clean`.

    Returns:
        str: Normalized, whitespace‑collapsed text ready for tokenization.
    """
    text = text.lower()
    # Keep numbers, currency symbols, punctuation - they're spam features!
    # Only remove truly problematic characters
    text = re.sub(r'[^\w\s£$€¥!?.,;:\'\"-]', ' ', text)
    text = re.sub(r'\s+', ' ', text)
    return text.strip()

print("[*] Applying minimal text cleaning (preserving spam indicators)...")
df['preprocessed'] = df['message'].apply(minimal_clean)

# Apply final cleaning for vectorization
df['clean_message'] = df['preprocessed'].apply(clean_text)

print("\n[*] Sample spam messages with preserved features:")
spam_samples = df[df['label'] == 'spam'].sample(3, random_state=42)
for idx, row in spam_samples.iterrows():
    msg = row['preprocessed'][:100] + "..." if len(row['preprocessed']) > 100 else row['preprocessed']
    print(f"  - {msg}")

# Remove only exact duplicates
original_size = len(df)
df = df.drop_duplicates(subset=['label', 'clean_message'])
print(f"\n[+] Removed {original_size - len(df)} duplicates")

# Remove empty messages
before_empty = len(df)
df = df[df['clean_message'].str.len() > 0]
print(f"[+] Removed {before_empty - len(df)} empty messages")

X = df['clean_message'].values
y = df['label'].values

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

print(f"\n[+] Data split:")
print(f"    Training: {len(X_train)} messages")
print(f"    Testing: {len(X_test)} messages")

print("\n[*] Training Naive Bayes classifier...")

model_dir = Path("models")
model_dir.mkdir(exist_ok=True)
model_path = model_dir / "spam_classifier.pkl"

if model_path.exists():
    print(f"[+] Loading saved model from {model_path}")
    with open(model_path, 'rb') as f:
        saved_data = pickle.load(f)
        vectorizer = saved_data['vectorizer']
        classifier = saved_data['classifier']

    # Transform data using existing vocabulary
    X_train_vec = vectorizer.transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
else:
    # Configure vectorizer to capture spam patterns
    vectorizer = CountVectorizer(
        max_features=3000,
        token_pattern=r'\b\w+\b|[£$€¥]+|\d+|!!+|\?\?+|\.\.+',
        lowercase=True,
        stop_words='english'
    )
    X_train_vec = vectorizer.fit_transform(X_train)
    X_test_vec = vectorizer.transform(X_test)
    classifier = MultinomialNB()
    classifier.fit(X_train_vec, y_train)

    # Save model for reproducibility
    with open(model_path, 'wb') as f:
        pickle.dump({'vectorizer': vectorizer, 'classifier': classifier}, f)
    print(f"[+] Model saved to {model_path}")

# Calculate accuracy scores
train_acc = classifier.score(X_train_vec, y_train)
test_acc = classifier.score(X_test_vec, y_test)
print(f"[+] Training accuracy: {train_acc:.4f}")
print(f"[+] Testing accuracy: {test_acc:.4f}")

# Get detailed predictions
y_pred = classifier.predict(X_test_vec)
print("\n[*] Classification Report:")
print(classification_report(y_test, y_pred))

print("\n[*] Extracting GoodWords from model...")

# Get feature names and probabilities
feature_names = vectorizer.get_feature_names_out()
ham_log_probs = classifier.feature_log_prob_[0]  # Ham class
spam_log_probs = classifier.feature_log_prob_[1]  # Spam class

# Calculate goodness scores
goodness_scores = []
for i, word in enumerate(feature_names):
    ham_prob = np.exp(ham_log_probs[i])
    spam_prob = np.exp(spam_log_probs[i])
    goodness = ham_prob / (spam_prob + 1e-10)
    goodness_scores.append((word, goodness, ham_prob, spam_prob))

# Sort by goodness
goodness_scores.sort(key=lambda x: x[1], reverse=True)
top_good_words = goodness_scores[:100]

print(f"[+] Top 10 GoodWords (most 'ham-like'):")
for word, score, hp, sp in top_good_words[:10]:
    print(f"    {word:15} | goodness: {score:8.2f} | ham_p: {hp:.4f} | spam_p: {sp:.4f}")

print("\n[*] Testing GoodWords attack...")

# Extract only spam messages for testing
spam_test_messages = X_test[y_test == 'spam']
print(f"[+] Testing on {len(spam_test_messages)} spam messages")

# Define test points from baseline (0) to saturation (40)
word_counts = [0, 5, 10, 15, 20, 25, 30, 35, 40]
attack_results = []

print(f"[*] Testing word counts: {word_counts}")

for num_words in word_counts:
    # Select the top N good words for this iteration
    selected_words = [w for w, _, _, _ in top_good_words[:num_words]]

    # Show which words we're using (first iteration only for clarity)
    if num_words == 5:
        print(f"  Using words: {', '.join(selected_words)}")

def augment_message(message, words_to_add):
    """Append good words to a message"""
    if len(words_to_add) > 0:
        return message + " " + " ".join(words_to_add)
    return message

# Test augmentation on one example using the top 5 words
sample_spam = spam_test_messages[0]
sample_augmented = augment_message(
    sample_spam,
    [w for w, _, _, _ in top_good_words[:5]]
)
print(f"\nOriginal: {sample_spam[:50]}...")
print(f"Augmented: {sample_augmented[:80]}...")

for num_words in word_counts:
    # Select the top N good words for this iteration
    selected_words = [w for w, _, _, _ in top_good_words[:num_words]]

    # Count how many spam messages evade after augmentation
    evaded = 0
    for message in spam_test_messages:
        # Augment the message
        augmented = augment_message(message, selected_words)

        # Transform and predict
        vec = vectorizer.transform([augmented])
        prob = classifier.predict_proba(vec)[0]

        # Check evasion: ham probability > spam probability
        if prob[0] > prob[1]:
            evaded += 1

    # Record results for this configuration
    evasion_rate = (evaded / len(spam_test_messages)) * 100
    attack_results.append({
        'num_words': num_words,
        'evasion_rate': evasion_rate,
        'evaded': evaded,
        'total': len(spam_test_messages)
    })

    print(f"  Words: {num_words:2d} | Evasion: {evasion_rate:6.2f}% ({evaded}/{len(spam_test_messages)})")

# Convert to DataFrame for easy plotting
results_df = pd.DataFrame(attack_results)

plt.style.use('dark_background')
fig, ax = plt.subplots(figsize=(12, 6), facecolor=NODE_BLACK)

ax.plot(results_df['num_words'], results_df['evasion_rate'],
        marker='o', markersize=8, linewidth=2.5,
        color=HTB_GREEN, markeredgecolor='white', markeredgewidth=1)

ax.fill_between(results_df['num_words'], 0, results_df['evasion_rate'],
                alpha=0.3, color=HTB_GREEN)

# Add threshold lines
ax.axhline(y=50, color=NUGGET_YELLOW, linestyle='--', alpha=0.7, label='50% threshold')
ax.axhline(y=90, color=AZURE, linestyle='--', alpha=0.7, label='90% threshold')

# Highlight maximum
max_idx = results_df['evasion_rate'].idxmax()
max_rate = results_df.loc[max_idx, 'evasion_rate']
max_words = results_df.loc[max_idx, 'num_words']
ax.scatter(max_words, max_rate, s=200, color=MALWARE_RED, zorder=5)
ax.annotate(f'Peak: {max_rate:.1f}%\n@ {max_words} words',
           xy=(max_words, max_rate), xytext=(max_words+5, max_rate-10),
           color='white', fontsize=10,
           arrowprops=dict(arrowstyle='->', color=MALWARE_RED, lw=1.5))

ax.set_xlabel('Number of Good Words Added', fontsize=12, color=HTB_GREEN)
ax.set_ylabel('Evasion Rate (%)', fontsize=12, color=HTB_GREEN)
ax.set_title('GoodWords Attack Effectiveness', fontsize=14, color=HTB_GREEN, pad=20)
ax.grid(True, alpha=0.2)
ax.set_facecolor(NODE_BLACK)
ax.legend()

for spine in ax.spines.values():
    spine.set_color(HACKER_GREY)
ax.tick_params(colors=HACKER_GREY)

plt.tight_layout()
output_dir = Path("attachments")
output_dir.mkdir(exist_ok=True)
plt.savefig(output_dir / "attack_effectiveness.png", dpi=150, facecolor=NODE_BLACK)
plt.close()
print(f"\n[+] Plot saved to {output_dir / 'attack_effectiveness.png'}")