import pandas as pd
import random
from nltk import FreqDist
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer

stopWords = set(stopwords.words('english'))
ps = PorterStemmer()

filename = r"D:\Svalbard\Data\AmazonMLChallengeData\dataset\train.csv" 
df = pd.read_csv(filename, nrows=50000, escapechar="\\", quoting = 3)
pd.set_option("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 200)
# node_grp = df.groupby('BROWSE_NODE_ID', axis='index)
for i in range(10):
    corpus = df.loc[df['BROWSE_NODE_ID'] == i]['DESCRIPTION'].str.cat(sep=' ')
    corpus_no_punch = corpus.translate(str.maketrans('', '', string.punctuation))
    corpus_no_stop_no_punch = ''
    for w in corpus_no_punch.split():
        if w.lower() not in stopWords and len(w) > 3:
            corpus_no_stop_no_punch += ps.stem(w) + ' '

    tokens = nltk.word_tokenize(corpus_no_stop_no_punch)
    fdist = FreqDist(tokens)
    print(i)
    print(sorted(fdist.items(), key=lambda k: k[1], reverse=True)[:10])
    print()