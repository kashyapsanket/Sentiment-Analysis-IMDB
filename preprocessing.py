from collections import Counter
from string import punctuation
import nltk
from nltk.corpus import stopwords
from os import listdir
import pandas as pd

nltk.download('stopwords')
reviews = []
labels = []


def clean(text):
    tokens = text.split()
    table = str.maketrans('', '', punctuation)
    tokens = [w.translate(table) for w in tokens]
    tokens = [word for word in tokens if word.isalpha()]
    stop_words = set(stopwords.words('english'))
    tokens = [w for w in tokens if not w in stop_words]
    tokens = [w for w in tokens if len(w) > 2]
    tokens = [w.lower() for w in tokens]
    return tokens


def clean_update1(text, vocab, label):
    tokens = clean(text)
    vocab.update(tokens)
    reviews.append(" ".join(tokens))
    labels.append(label)


def file_reader(list_dir, vocab):
    for dir in list_dir:
        for file in listdir(dir):
            #print(file)
            if not file.endswith(".txt"):
                continue
            curr_file = dir + '/' + file
            f = open(curr_file, 'r')
            text = f.read()
            f.close()
            label = 0
            if dir == 'reviews/train/pos' or dir == 'reviews/test/pos':
                label = 1
            clean_update1(text, vocab, label)


if __name__ == "__main__":
    train_dir = ['reviews/train/pos','reviews/train/neg']
    test_dir = ['reviews/test/pos', 'reviews/test/neg']
    vocab = Counter()
    headings = ['Series', 'Reviews', 'Labels']

    file_reader(train_dir, vocab)
    training_set = pd.DataFrame(list(zip(reviews,labels)))
    training_set.to_csv('TrainingSet.csv')

    reviews[:] = []
    labels[:] = []

    test_vocab = Counter()
    file_reader(test_dir, test_vocab)
    test_set = pd.DataFrame(list(zip(reviews, labels)))
    test_set.to_csv('TestSet.csv')



