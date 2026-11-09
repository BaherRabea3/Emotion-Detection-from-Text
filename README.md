# Emotion Detection from Text

A Natural Language Processing (NLP) project that aims to identify emotions expressed in text, such as joy, sadness, anger, and fear.

The project explores text preprocessing and statistical language modeling as part of the emotion detection workflow.

## Project Objective

The main objective is to analyze textual data and prepare it for emotion detection by applying NLP preprocessing techniques and exploring word sequence probabilities using N-gram language models.

##  Dataset

**Dataset:** [Emotion Detection from Text – Kaggle](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)

The dataset contains tweets labeled with different emotions.

## 🛠️ Project Workflow

### 1. Text Preprocessing

The preprocessing stage includes:

- Removing missing and duplicate records.
- Converting text to lowercase.
- Removing URLs and user mentions.
- Handling contractions.
- Tokenization using NLTK TweetTokenizer.
- Removing punctuation and numbers.
- Removing stop words while preserving important negation words.
- Lemmatization using WordNet Lemmatizer.

The cleaned dataset is saved as `tweets_clean.csv`.

### 2. N-gram Language Modeling

A Bigram Language Model is implemented to calculate the probability of word sequences using the Maximum Likelihood Estimation (MLE) approach.

**Bigram Probability Formula:**

\[
P(w_i \mid w_{i-1}) =
\frac{Count(w_{i-1},w_i)}{Count(w_{i-1})}
\]

The model uses the Markov assumption, where each word depends only on the previous word.

The implementation includes:

- Counting unigrams and bigrams.
- Adding sentence boundary markers (`<s>` and `</s>`).
- Calculating conditional probabilities for consecutive words.
- Calculating the probability of a complete tweet.
- Calculating log probability to represent very small probabilities.
- Displaying dataset statistics.

## 📁 Project Structure

```text
Emotion-Detection-from-Text/
│
├── ngram_analysis.py
├── tweets_clean.csv
├── README.md
└── ...
```

*Other project files, including the preprocessing script and original dataset, are maintained according to the repository structure.*

## Requirements

- Python 3
- Pandas
- NLTK

Install the required libraries:

```bash
pip install pandas nltk
```

##  How to Run

1. Run the preprocessing script to generate `tweets_clean.csv`.
2. Make sure the cleaned dataset is available in the same directory as `ngram_analysis.py`.
3. Run the Bigram analysis:

```bash
python ngram_analysis.py
```

The script displays the selected tweet, individual bigram probability calculations, the complete sentence probability, log probability, and dataset statistics.

##  Current Scope

The current implementation covers text preprocessing and Bigram language modeling using MLE. These steps form part of the broader Emotion Detection project workflow.



- [Emotion Detection Dataset – Kaggle](https://www.kaggle.com/datasets/pashupatigupta/emotion-detection-from-text)
- NLTK Documentation: https://www.nltk.org/
