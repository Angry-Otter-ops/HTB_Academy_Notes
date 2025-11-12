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

print("\n[*] Analyzing individual word impact...")

# Use 50 spam messages as a representative sample
sample_spam = spam_test_messages[:50]
word_impacts = []

print(f"[+] Testing {len(top_good_words[:20])} words on {len(sample_spam)} spam messages")

for word, _, _, _ in top_good_words[:20]:
    total_impact = 0

    for message in sample_spam:
        # Calculate original spam probability
        vec_orig = vectorizer.transform([message])
        prob_orig = classifier.predict_proba(vec_orig)[0][1]  # spam prob

        # Calculate probability after adding the word
        vec_aug = vectorizer.transform([message + " " + word])
        prob_aug = classifier.predict_proba(vec_aug)[0][1]

        # Measure the probability reduction
        impact = prob_orig - prob_aug
        total_impact += impact
    
    # Calculate average impact across all messages
    avg_impact = (total_impact / len(sample_spam)) * 100
    word_impacts.append((word, avg_impact))

    # Show progress for first few words
    if len(word_impacts) <= 3:
        print(f"  Word '{word}': {avg_impact:.2f}% reduction")

word_impacts.sort(key=lambda x: x[1], reverse=True)

# Plot word impacts
fig, ax = plt.subplots(figsize=(10, 8), facecolor=NODE_BLACK)

words = [w for w, _ in word_impacts]
impacts = [i for _, i in word_impacts]
colors = [HTB_GREEN if i > 15 else NUGGET_YELLOW if i > 10 else HACKER_GREY for i in impacts]

bars = ax.barh(range(len(words)), impacts, color=colors, edgecolor='white', linewidth=0.5)

ax.set_yticks(range(len(words)))
ax.set_yticklabels(words)
ax.set_xlabel('Average Spam Probability Reduction (%)', fontsize=12, color=HTB_GREEN)
ax.set_title('Individual Word Impact on Spam Detection', fontsize=14, color=HTB_GREEN, pad=20)
ax.grid(axis='x', alpha=0.2)
ax.set_facecolor(NODE_BLACK)

for spine in ax.spines.values():
    spine.set_color(HACKER_GREY)
ax.tick_params(colors=HACKER_GREY)

plt.tight_layout()
plt.savefig(output_dir / "word_impact.png", dpi=150, facecolor=NODE_BLACK)
plt.close()
print(f"[+] Plot saved to {output_dir / 'word_impact.png'}")

print("\n[*] Visualizing probability shifts...")

# Sample messages for detailed analysis
sample_messages = spam_test_messages[:8]
test_word_counts = [0, 5, 10, 20, 30]

fig, ax = plt.subplots(figsize=(14, 6), facecolor=NODE_BLACK)

x = np.arange(len(sample_messages))
width = 0.15
colors_list = [MALWARE_RED, NUGGET_YELLOW, AZURE, HTB_GREEN, AQUAMARINE]

for i, num_words in enumerate(test_word_counts):
    # Select the appropriate number of good words
    selected = [w for w, _, _, _ in top_good_words[:num_words]]
    probs = []

    for msg in sample_messages:
        # Augment message with selected words
        if num_words > 0:
            aug_msg = msg + " " + " ".join(selected)
        else:
            aug_msg = msg

        # Calculate spam probability
        vec = vectorizer.transform([aug_msg])
        spam_prob = classifier.predict_proba(vec)[0][1]
        probs.append(spam_prob)

    # Create grouped bars with distinct colors
    bars = ax.bar(x + i*width, probs, width,
                   label=f'{num_words} words',
                   color=colors_list[i], alpha=0.8)
    
    # Mark successful evasions
    for j, (bar, prob) in enumerate(zip(bars, probs)):
        if prob < 0.5:
            ax.text(bar.get_x() + bar.get_width()/2, prob + 0.02,
                   '✓', ha='center', va='bottom', color=HTB_GREEN, fontweight='bold')
ax.axhline(y=0.5, color='white', linestyle='--', alpha=0.5, label='Decision boundary')
ax.set_xlabel('Message Index', fontsize=12, color=HTB_GREEN)
ax.set_ylabel('Spam Probability', fontsize=12, color=HTB_GREEN)
ax.set_title('Probability Shift with Increasing Good Words', fontsize=14, color=HTB_GREEN, pad=20)
ax.set_xticks(x + width * 2)
ax.set_xticklabels([f'M{i+1}' for i in range(len(sample_messages))])
ax.legend(loc='upper right')
ax.grid(axis='y', alpha=0.2)
ax.set_facecolor(NODE_BLACK)

for spine in ax.spines.values():
    spine.set_color(HACKER_GREY)
ax.tick_params(colors=HACKER_GREY)

plt.tight_layout()
plt.savefig(output_dir / "probability_shift.png", dpi=150, facecolor=NODE_BLACK)
plt.close()
print(f"[+] Plot saved to {output_dir / 'probability_shift.png'}")

print("\n[*] Simulating black-box attack scenario...")
print("[*] Budget: 1000 queries")

# Simulate limited query access
query_budget = 1000
queries_used = 0
query_log = []

def extract_ham_word_freq(X_train, y_train, sample_size=500):
    """
    Compute token frequencies from a sample of ham messages.

    Parameters
    ----------
    X_train : array-like of str
        Cleaned training messages.
    y_train : array-like of str
        Labels aligned with X_train ('ham' or 'spam').
    sample_size : int, default 500
        Number of ham messages to analyze.

    Returns
    -------
    dict[str, int]
        Mapping of word -> frequency within sampled ham messages.
    """
    ham_msgs = X_train[y_train == 'ham']
    limit = min(sample_size, len(ham_msgs))
    freq = {}
    for msg in ham_msgs[:limit]:
        for w in str(msg).split():
            if 2 < len(w) < 10:  # keep typical conversational tokens
                freq[w] = freq.get(w, 0) + 1
    return freq

wf_example = extract_ham_word_freq(X_train, y_train, sample_size=500)
print("[*] Example: extract_ham_word_freq")
print(f"  Ham messages sampled: {min(500, sum(y_train == 'ham'))}")
print(f"  Unique tokens found: {len(wf_example)}")
top5 = sorted(wf_example.items(), key=lambda x: (-x[1], x[0]))[:5]
for w, c in top5:
    print(f"    {w}: {c}")

def select_high_frequency_words(word_freq, max_words=100, min_freq=5):
    """
    Select the most frequent ham words above a minimum frequency.

    Parameters
    ----------
    word_freq : dict[str, int]
        Token frequency table for sampled ham messages.
    max_words : int, default 100
        Maximum number of words to return.
    min_freq : int, default 5
        Minimum frequency a word must meet to be considered.

    Returns
    -------
    list[str]
        Top words sorted by decreasing frequency then lexicographically.
    """
    sorted_by_freq = sorted(word_freq.items(), key=lambda x: (-x[1], x[0]))
    top = [w for w, c in sorted_by_freq if c > min_freq][:max_words]
    return top

top_words_example = select_high_frequency_words(wf_example, max_words=100, min_freq=5)
print("[*] Example: select_high_frequency_words")
print(f"  Selected top words: {len(top_words_example)} (min_freq=5)")
print("  First 10:", ", ".join(top_words_example[:10]))

def merge_with_curated(top_words, additional_candidates=None):
    """
    Merge data-driven top words with curated conversational candidates.

    Parameters
    ----------
    top_words : list[str]
        High-frequency ham words from the previous step.
    additional_candidates : list[str] | None
        Optional curated list to include regardless of frequency.

    Returns
    -------
    list[str]
        Deduplicated merged list (lexicographically ordered).
    """
    if additional_candidates is None:
        additional_candidates = [
            "ok", "cos", "ill", "thats", "later", "said", "ask", "didnt",
            "dont", "doing", "going", "come", "home", "tomorrow", "today", "sorry",
            "thanks", "yeah", "yes", "sure", "see", "tell", "know", "think",
        ]
    merged = set(top_words) | set(additional_candidates)
    return sorted(merged)

merged_example = merge_with_curated(top_words_example)
added = sorted(set(merged_example) - set(top_words_example))
print("[*] Example: merge_with_curated")
print(f"  Merged size: {len(merged_example)} | Added curated: {len(added)}")
print("  Sample added terms:", ", ".join(added[:5]))

def build_candidate_vocabulary(
    X_train,
    y_train,
    sample_size=500,
    max_words=100,
    min_freq=5,
    additional_candidates=None,
):
    """
    Build a candidate vocabulary for black-box discovery from ham messages.

    Parameters
    ----------
    X_train : array-like of str
        Cleaned training messages.
    y_train : array-like of str
        Labels aligned with X_train ('ham' or 'spam').
    sample_size : int, default 500
        Number of ham messages to analyze.
    max_words : int, default 100
        Maximum number of top frequent ham words to keep before merging extras.
    min_freq : int, default 5
        Minimum frequency threshold for inclusion from the ham corpus.
    additional_candidates : list[str] | None
        Optional curated conversational terms to include.

    Returns
    -------
    list[str]
        Deduplicated candidate words ordered by decreasing ham frequency,
        then lexicographically for stable ties.
    """
    word_freq = extract_ham_word_freq(X_train, y_train, sample_size=sample_size)
    top_words = select_high_frequency_words(word_freq, max_words=max_words, min_freq=min_freq)
    merged = merge_with_curated(top_words, additional_candidates=additional_candidates)

    # Stable final ordering driven by ham frequency, then lexical for ties
    def sort_key(w):
        return (-word_freq.get(w, 0), w)

    return sorted(merged, key=sort_key)

cv_example = build_candidate_vocabulary(X_train, y_train)
print("[*] Example: build_candidate_vocabulary")
print(f"  Candidates: {len(cv_example)}")
print("  First 10:", ", ".join(cv_example[:10]))

# Build candidate vocabulary for discovery
candidate_words = build_candidate_vocabulary(X_train, y_train)
print(f"[+] Testing {len(candidate_words)} candidate words extracted from ham messages")

def estimate_budget_allocation(total_budget):
    """
    Estimate allocation across exploration, exploitation, and combination.

    Parameters
    ----------
    total_budget : int
        Total query budget available for discovery.

    Returns
    -------
    dict
        Mapping phase -> integer number of queries that sums to `total_budget`.
    """
    explore = int(0.4 * total_budget)
    exploit = int(0.4 * total_budget)
    combine = total_budget - explore - exploit  # absorb rounding
    return {
        'exploration': explore,
        'exploitation': exploit,
        'combination': combine,
    }

# Quick demo for budget allocation
allocation = estimate_budget_allocation(query_budget)
print("\n[*] Budget allocation:")
for phase, budget in allocation.items():
    print(f"  {phase:12}: {budget:4d} queries")
print(f"  Total: {sum(allocation.values())} / {query_budget}")

# Discovery phase - test word effectiveness
word_scores = {}
test_spam_samples = spam_test_messages[:50]  # More test messages

# Test in batches to be more efficient
print(f"[*] Discovery phase: testing {len(candidate_words)} candidates...")

# Randomly sample candidates and messages for better coverage
np.random.shuffle(candidate_words)
np.random.shuffle(test_spam_samples)

def initialize_adaptive_scorer():
    """Initialize adaptive scoring data structures"""
    return {
        'word_scores': {},      # Maps word -> effectiveness score
        'word_counts': {},      # Maps word -> number of times tested
        'exploration_rate': 0.2  # 20% exploration for discovery phase
    }

def epsilon_greedy_select(scorer, available_words):
    """Select word using epsilon-greedy strategy

    Parameters:
        scorer (dict): Adaptive scorer state
        available_words (list): Candidate words to choose from

    Returns:
        str: Selected word for testing
    """
    import random

    if random.random() < scorer['exploration_rate']:
        # Exploration: try untested or rarely tested words
        untested = [w for w in available_words if w not in scorer['word_counts']]
        if untested:
            return random.choice(untested)
        else:
            # Choose least tested word
            return min(available_words,
                      key=lambda w: scorer['word_counts'].get(w, 0))
    else:
        # Exploitation: choose best performing word
        return max(available_words,
                  key=lambda w: scorer['word_scores'].get(w, 0))
def update_word_score(scorer, word, impact, alpha=0.3):
    """Update word score using exponential moving average

    Parameters:
        scorer (dict): Adaptive scorer state
        word (str): Word being scored
        impact (float): Observed reduction in spam probability
        alpha (float): Learning rate
    """
    if word not in scorer['word_scores']:
        scorer['word_scores'][word] = impact
        scorer['word_counts'][word] = 1
    else:
        # Exponential moving average
        old_score = scorer['word_scores'][word]
        scorer['word_scores'][word] = (1 - alpha) * old_score + alpha * impact
        scorer['word_counts'][word] += 1
def discover_word_combinations(message, test_words, max_size=3):
    """Discover effective word combinations through systematic search

    Parameters:
        message (str): Target spam message
        test_words (list): Promising words to test
        max_size (int): Maximum combination size

    Returns:
        dict: Mapping of word combinations to effectiveness scores
    """
    from itertools import combinations

    combination_scores = {}
    message_vec = vectorizer.transform([message])
    message_score = classifier.predict_proba(message_vec)[0][1]
    # Test individual words first
    for word in test_words[:20]:
        test_message = message + " " + word
        test_vec = vectorizer.transform([test_message])
        score = classifier.predict_proba(test_vec)[0][1]
        impact = message_score - score
        combination_scores[(word,)] = impact
    
    # Test pairs for synergy
    if max_size >= 2:
        for word1, word2 in combinations(test_words[:15], 2):
            test_message = message + " " + word1 + " " + word2
            test_vec = vectorizer.transform([test_message])
            score = classifier.predict_proba(test_vec)[0][1]

            # Calculate synergy
            individual_impact = combination_scores.get((word1,), 0) + combination_scores.get((word2,), 0)
            actual_impact = message_score - score
            synergy = actual_impact - individual_impact

            if synergy > 0:  # Positive synergy detected
                combination_scores[(word1, word2)] = actual_impact
   
    # Test triplets for top pairs
    if max_size >= 3:
        top_pairs = sorted(
            [(k, v) for k, v in combination_scores.items() if len(k) == 2],
            key=lambda x: x[1], reverse=True
        )[:5]

        for pair, pair_score in top_pairs:
            for word in test_words[:10]:
                if word not in pair:
                    triplet = tuple(sorted(pair + (word,)))
                    test_message = message + " " + " ".join(triplet)
                    test_vec = vectorizer.transform([test_message])
                    score = classifier.predict_proba(test_vec)[0][1]
                    combination_scores[triplet] = message_score - score

    return combination_scores

def three_phase_discovery(spam_messages, candidate_words, budget=1000):
    """Three-phase discovery: exploration, exploitation, combination

    Parameters:
        spam_messages (list): Target spam messages
        candidate_words (list): Vocabulary to test
        budget (int): Total query budget

    Returns:
        tuple: (discovered_words, combination_scores, queries_used)
    """
    scorer = initialize_adaptive_scorer()
    queries_used = 0

    # Allocate budgets using 40-40-20 split strategy
    allocation = estimate_budget_allocation(budget)
    exploration_budget = allocation['exploration']
    exploitation_budget = allocation['exploitation']
    combination_budget = allocation['combination']
    # Phase 1: Broad exploration (allocated budget)
    print(f"[*] Phase 1: Exploration (budget: {exploration_budget} queries)")

    p1_marks = {
        max(1, int(0.25 * exploration_budget)),
        max(1, int(0.50 * exploration_budget)),
        max(1, int(0.75 * exploration_budget)),
    }
    p1_reported = set()
    
    # Select a message and a candidate word
    test_message = random.choice(spam_messages)
    word = epsilon_greedy_select(scorer, candidate_words)

    # Baseline and augmented spam probabilities
    vec_orig = vectorizer.transform([test_message])
    prob_orig = classifier.predict_proba(vec_orig)[0][1]  # spam prob

    vec_aug = vectorizer.transform([test_message + " " + word])
    prob_aug = classifier.predict_proba(vec_aug)[0][1]

    impact = prob_orig - prob_aug

    # Update running score and consume query budget
    update_word_score(scorer, word, impact)
    queries_used += 2

        # Baseline and augmented spam probabilities
    vec_orig = vectorizer.transform([test_message])
    prob_orig = classifier.predict_proba(vec_orig)[0][1]  # spam prob

    vec_aug = vectorizer.transform([test_message + " " + word])
    prob_aug = classifier.predict_proba(vec_aug)[0][1]

    impact = prob_orig - prob_aug

    # Update running score and consume query budget
    update_word_score(scorer, word, impact)
    queries_used += 2

    # Optional milestone report
    if queries_used in p1_marks and queries_used not in p1_reported:
        top3 = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
        print(
            f"  [P1 {queries_used}/{exploration_budget}] "
            f"tested_words={len(scorer['word_scores'])} | "
            f"top3=" + ", ".join(f"{w}:{s:.3f}" for w, s in top3)
        )
        p1_reported.add(queries_used)

    while queries_used < exploration_budget and len(candidate_words) > 0:
        # Select inputs
        test_message = random.choice(spam_messages)
        word = epsilon_greedy_select(scorer, candidate_words)

        # Measure impact with two queries
        vec_orig = vectorizer.transform([test_message])
        prob_orig = classifier.predict_proba(vec_orig)[0][1]
        vec_aug = vectorizer.transform([test_message + " " + word])
        prob_aug = classifier.predict_proba(vec_aug)[0][1]
        impact = prob_orig - prob_aug

        # Update score and account for budget
        update_word_score(scorer, word, impact)
        queries_used += 2

        # Milestone report
        if queries_used in p1_marks and queries_used not in p1_reported:
            top3 = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)[:3]
            print(
                f"  [P1 {queries_used}/{exploration_budget}] "
                f"tested_words={len(scorer['word_scores'])} | "
                f"top3=" + ", ".join(f"{w}:{s:.3f}" for w, s in top3)
            )
            p1_reported.add(queries_used)

    print(f"[+] Exploration complete. Queries: {queries_used}, Words tested: {len(scorer['word_scores'])}")
    top5 = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)[:5]
    if top5:
        print("  Top5 after exploration:")
        for w, s in top5:
            print(f"    {w:12} | score: {s:.3f}")

    # Phase 2: Focused exploitation
    scorer['exploration_rate'] = 0.1  # Reduce exploration

    # Get top words for exploitation
    top_words = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)[:30]
    top_word_list = [w for w, _ in top_words]

    print(f"\n[*] Phase 2: Exploitation (budget: {exploitation_budget} queries)")
    initial_queries = queries_used
    p2_mid = initial_queries + max(1, exploitation_budget // 2)

    while queries_used < initial_queries + exploitation_budget and len(top_word_list) > 0:
        test_message = random.choice(spam_messages[:20])  # Focus on fewer messages
        word = random.choice(top_word_list[:15])  # Focus on best words

        vec_orig = vectorizer.transform([test_message])
        prob_orig = classifier.predict_proba(vec_orig)[0][1]

        vec_aug = vectorizer.transform([test_message + " " + word])
        prob_aug = classifier.predict_proba(vec_aug)[0][1]

        impact = prob_orig - prob_aug
        update_word_score(scorer, word, impact)
        queries_used += 2

    print(f"[+] Exploitation complete. Total queries: {queries_used}")

        # Phase 3: Combination discovery (allocated budget)
    remaining_combo = combination_budget
    print(f"\n[*] Phase 3: Combination search (budget: {remaining_combo} queries)")

    best_combinations = {}
    combos_tested = 0

    if remaining_combo > 50:  # Need minimum queries for combinations
        for i in range(min(3, len(spam_messages))):
            if queries_used >= budget or remaining_combo <= 0:
                break

            test_msg = spam_messages[i]
            combos = discover_word_combinations(test_msg, top_word_list[:20], max_size=3)

            # Track best combinations across messages
            for combo, score in combos.items():
                if combo not in best_combinations or score > best_combinations[combo]:
                    best_combinations[combo] = score

            # Account for queries (~2 per combination) while respecting the budget
            to_add = min(remaining_combo, len(combos) * 2)
            queries_used += to_add
            remaining_combo -= to_add
            combos_tested += len(combos)

            # Midpoint snapshot
            if combination_budget > 0 and remaining_combo <= combination_budget // 2 and best_combinations:
                best = max(best_combinations.items(), key=lambda x: x[1])
                print(
                    f"  [P3 mid ~{combination_budget - remaining_combo}/{combination_budget}] "
                    f"combos_tested={combos_tested} | best={' + '.join(best[0])}:{best[1]:.3f}"
                )

            if remaining_combo <= 0:
                break
    print(f"[+] Combination search complete. Total queries: {queries_used}")

    # Return final results
    final_words = sorted(scorer['word_scores'].items(), key=lambda x: x[1], reverse=True)
    return final_words, best_combinations, queries_used

print("\n[*] Using three-phase discovery algorithm...")

# Build candidate vocabulary
candidate_words = build_candidate_vocabulary(X_train, y_train)
print(f"[+] Built vocabulary of {len(candidate_words)} candidate words")

# Show budget allocation
allocation = estimate_budget_allocation(query_budget)
print(f"\n[*] Budget allocation:")
for phase, budget in allocation.items():
    print(f"    {phase:12}: {budget:4d} queries")

# Run three-phase discovery
discovered_words, combination_scores, total_queries = three_phase_discovery(
    spam_test_messages[:50],
    candidate_words,
    budget=query_budget
)

print(f"\n[+] Discovery complete. Total queries used: {total_queries}/{query_budget}")
print(f"[+] Top 10 discovered words:")
for word, score in discovered_words[:10]:
    print(f"    {word:10} | impact: {score:.3f}")

if combination_scores:
    print(f"\n[+] Top 5 word combinations:")
    top_combos = sorted(combination_scores.items(), key=lambda x: x[1], reverse=True)[:5]
    for combo, score in top_combos:
        combo_str = ', '.join(combo)
        print(f"    {combo_str:30} | synergy: {score:.3f}")

# Update queries_used for compatibility
queries_used = total_queries

# Test discovered words
blackbox_results = []
test_counts = [0, 5, 10, 15, 20, 25, 30]  # Test with more words

for num_words in test_counts:
    if queries_used >= query_budget:
        break

    selected = [w for w, _ in discovered_words[:num_words]]
    evaded = 0
    tested = 0

    # Test on a different subset of spam messages
    eval_messages = spam_test_messages[30:50]  # Different messages from discovery

    for msg in eval_messages:
        if queries_used >= query_budget:
            break

        aug = msg if num_words == 0 else msg + " " + " ".join(selected)
        vec = vectorizer.transform([aug])
        prob = classifier.predict_proba(vec)[0][1]  # spam prob
        queries_used += 1

        if prob < 0.5:  # evasion threshold
            evaded += 1
        tested += 1

    if tested > 0:
        rate = (evaded / tested) * 100
        blackbox_results.append({'num_words': num_words, 'evasion_rate': rate})
        print(f"  Words: {num_words:2d} | Evasion: {rate:6.2f}% | Queries total: {queries_used}")

print(f"\n[+] Black-box attack complete. Total queries: {queries_used}/{query_budget}")

print("\n" + "="*60)
print("ATTACK SUMMARY")
print("="*60)
print(f"Model Accuracy: {test_acc:.2%}")
print(f"Best White-box Evasion: {results_df['evasion_rate'].max():.1f}% @ {results_df.loc[results_df['evasion_rate'].idxmax(), 'num_words']} words")
if blackbox_results:
    bb_max = max(r['evasion_rate'] for r in blackbox_results)
    print(f"Best Black-box Evasion: {bb_max:.1f}% (with {queries_used} queries)")
print(f"Most Effective Word: '{word_impacts[0][0]}' (reduces spam prob by {word_impacts[0][1]:.1f}%)")

# Save results
results = {
    'white_box': attack_results,
    'black_box': blackbox_results,
    'top_good_words': [(w, float(s)) for w, s, _, _ in top_good_words[:20]],
    'word_impacts': word_impacts[:10]
}

with open(output_dir / "results.json", 'w') as f:
    json.dump(results, f, indent=2)

print(f"\n[+] Results saved to {output_dir / 'results.json'}")
print("[+] Attack demonstration complete!")
