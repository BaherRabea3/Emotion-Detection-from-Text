
import pandas as pd
from collections import Counter
from math import prod, log

# 1. Load the cleaned dataset generated from the preprocessing step
df = pd.read_csv("tweets_clean.csv")

# Keep only valid cleaned tweets
df["clean_text"] = df["clean_text"].fillna("").astype(str)
df = df[df["clean_text"].str.strip() != ""]

# 2. Convert each tweet into a list of words
sentences = [
    text.split()
    for text in df["clean_text"]
]

# Check if the dataset contains valid tweets
if not sentences:
    raise ValueError("No valid tweets found in the cleaned dataset.")

# 3. Add sentence boundary markers
# <s>  = Start of sentence
# </s> = End of sentence
tokenized_sentences = [
    ["<s>"] + sentence + ["</s>"]
    for sentence in sentences
]

# 4. Count Unigrams and Bigrams
unigram_counts = Counter()
bigram_counts = Counter()

for sentence in tokenized_sentences:
    for i in range(len(sentence) - 1):
        previous_word = sentence[i]
        current_word = sentence[i + 1]

        unigram_counts[previous_word] += 1
        bigram_counts[(previous_word, current_word)] += 1

# 5. Calculate Bigram Probabilities using Maximum Likelihood Estimation (MLE)
# P(current_word | previous_word) =
# Count(previous_word, current_word) / Count(previous_word)

def bigram_probability(previous_word, current_word):
    bigram_count = bigram_counts[(previous_word, current_word)]
    previous_word_count = unigram_counts[previous_word]

    if previous_word_count == 0:
        return 0.0

    return bigram_count / previous_word_count


# 6. Select an existing tweet from the dataset
# This tweet will be used to demonstrate the probability calculations.

example = next(
    (s for s in sentences if 3 <= len(s) <= 8),
    sentences[0]
)

# 7. Calculate the probability of the whole tweet
# Using the Bigram Markov Assumption:
# Each word depends only on the previous word.

example_with_markers = ["<s>"] + example + ["</s>"]

probabilities = []

print("========== BIGRAM CALCULATION ==========")

print("\nSelected Tweet:")
print(" ".join(example))

print("\nBigram Probability Calculations:\n")

for i in range(len(example_with_markers) - 1):
    previous_word = example_with_markers[i]
    current_word = example_with_markers[i + 1]

    bigram_count = bigram_counts[(previous_word, current_word)]
    previous_count = unigram_counts[previous_word]

    probability = bigram_probability(previous_word, current_word)
    probabilities.append(probability)

    # Display the calculation steps for each bigram
    print(f"P({current_word} | {previous_word})")
    print(
        f"= Count({previous_word}, {current_word})"
        f" / Count({previous_word})"
    )
    print(f"= {bigram_count} / {previous_count}")
    print(f"= {probability:.6f}\n")


# 8. Calculate the complete tweet probability
# Sentence probability = Product of all consecutive bigram probabilities

tweet_probability = prod(probabilities)

print("========== FINAL RESULT ==========")

print("Tweet:", " ".join(example))

print("\nSentence Probability:")
print("P(Tweet) = " + " * ".join(
    f"{p:.6f}" for p in probabilities
))

print(f"\nP(Tweet) = {tweet_probability:.12g}")

# Log Probability helps represent very small probabilities
# without numerical underflow.

if any(p == 0 for p in probabilities):
    log_probability = float("-inf")
else:
    log_probability = sum(log(p) for p in probabilities)

print(f"Log Probability = {log_probability:.6f}")


# 9. Display dataset statistics

print("\n========== DATASET STATISTICS ==========")

print("Number of tweets:", len(sentences))
print("Total Unigram counts:", sum(unigram_counts.values()))
print("Total Bigram counts:", sum(bigram_counts.values()))
print("Unique Bigrams:", len(bigram_counts))
