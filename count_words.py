import string
from collections import Counter
import os
import pandas as pd
import matplotlib.pyplot as plt

book_dir = "./books"


def count_words(text):
    text = text.lower()
    skips = string.punctuation
    for ch in skips:
        text = text.replace(ch, '')
    word_counts = Counter(text.split(" "))
    return word_counts


def read_book(title_path):
    with open(title_path, "r", encoding="utf-8") as f:
        text = f.read()
        text = text.replace("\n", "").replace("\r", "").replace("\t", "")
        return text


def word_stats(word_counts):
    num_unique = len(word_counts)
    counts = word_counts.values()
    return (num_unique, counts)


# text = read_book("romeo_and_juliet.txt")
# word_counts = count_words(text)
# num_unique, counts = word_stats(word_counts)
# print(num_unique, sum(counts))

table = pd.DataFrame(columns=["Language", "Author", "Title", "Unique", "Words"])

t = 1
for language in os.listdir(book_dir):
    for author in os.listdir(os.path.join(book_dir, language)):
        for title in os.listdir(os.path.join(book_dir, language, author)):
            inputfile = os.path.join(book_dir, language, author, title)
            print(inputfile)
            text = read_book(inputfile)
            (num_unique, counts) = word_stats(count_words(text))
            table.loc[t] = [language.capitalize(), author.capitalize(), title.replace(".txt", ""), num_unique, sum(counts)]
            t += 1

print(table)

plt.figure(figsize=(10, 10))
subset = table[table["Language"] == "English"]
plt.loglog(subset.Words, subset.Unique, "o", label="English", color="crimson")
subset = table[table["Language"] == "German"]
plt.loglog(subset.Words, subset.Unique, "o", label="German", color="forestgreen")
plt.legend()
plt.xlabel("Words")
plt.ylabel("Unique Words")
plt.show()
