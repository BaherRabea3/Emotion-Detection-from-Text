# pip install pandas nltk
import re
import pandas as pd
import nltk
from nltk.tokenize import TweetTokenizer
from nltk.corpus import stopwords, wordnet
from nltk.stem import WordNetLemmatizer

# for pkg in ["stopwords", "wordnet", "omw-1.4",
#             "averaged_perceptron_tagger", "averaged_perceptron_tagger_eng"]:
#     nltk.download(pkg)

df = pd.read_csv("tweet_emotions\\tweet_emotions.csv")

# print(df.shape)
# print(df["sentiment"].value_counts())
# print(df.isna().sum())
# print("Duplicates:", df["content"].duplicated().sum())

df = df.dropna(subset=["content"])\
       .drop_duplicates(subset=["content"])\
       .reset_index(drop=True)

# print(df["content"].sample(10, random_state=1).to_string())

def clean_tweet(text):
    text = text.lower()
    text = re.sub(r"http\S+|www\.\S+", " ", text) #remove URLs
    text = re.sub(r"@\w+", " ", text) #remove mentions
    text = text.replace("#","")
    text = re.sub(r"(.)\1{2,}", r"\1\1", text) #gooooood ==> good
    text = text.replace("can't", "can not")
    text = text.replace("won't", "will not")
    text = re.sub(r"n't\b", " not", text)
    return text

tokenizer = TweetTokenizer(reduce_len=True)


def remove_punct_numbers(tokens):
    out = []
    for t in tokens:
        t = t.replace("'", "")        
        if t.isalpha():               
            out.append(t)
    return out


stop_words = set(stopwords.words("english"))
keep = {"not", "no", "nor", "never", "very", "too", "so", "against"}
stop_words = stop_words - keep

def remove_stopwords(tokens):
    return [t for t in tokens if t not in stop_words]


lemmatizer = WordNetLemmatizer()

def get_wn_pos(tag):
    if tag.startswith("J"): return wordnet.ADJ
    if tag.startswith("V"): return wordnet.VERB
    if tag.startswith("R"): return wordnet.ADV
    return wordnet.NOUN

def lemmatize(tokens):
    tagged = nltk.pos_tag(tokens)
    return [lemmatizer.lemmatize(w, get_wn_pos(t)) for w, t in tagged]

def preprocess(text):
    text = clean_tweet(text)
    tokens = tokenizer.tokenize(text)
    tokens = remove_punct_numbers(tokens)
    tokens = remove_stopwords(tokens)
    tokens = lemmatize(tokens)
    return " ".join(tokens)

df["clean_text"] = df["content"].apply(preprocess)

df = df[df["clean_text"].str.strip() != ""].reset_index(drop=True)
print(df[["content", "clean_text"]].head(10))

df.to_csv("tweets_clean.csv", index=False)
