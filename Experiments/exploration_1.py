import pandas as pd
import random
from nltk import FreqDist
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()

stopWords = set(stopwords.words('english'))
ps = PorterStemmer()
to_remove = string.punctuation + string.digits
filename = r"D:\Svalbard\Data\AmazonMLChallengeData\dataset\train.csv" 
df = pd.read_csv(filename, skiprows=0, nrows=10, escapechar="\\", quoting = 3)
# pd.set_option("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 200)
df['TITLE'] = df['TITLE'].apply(lambda line: ' '.join([lem.lemmatize(w.lower()) for w in line.translate(str.maketrans('', '', to_remove)).split() if w.lower() not in stopWords and len(w) > 3]))
print(df)
# node_grp = df.groupby('BROWSE_NODE_ID', axis='index)
# for i in range(10):
    # corpus = df.loc[df['BROWSE_NODE_ID'] == i]['DESCRIPTION'].str.cat(sep=' ')
    # corpus_no_punch = corpus.translate(str.maketrans('', '', string.punctuation))
    # corpus_no_stop_no_punch = ''
    # for w in corpus_no_punch.split():
        # if w.lower() not in stopWords and len(w) > 3:
            # corpus_no_stop_no_punch += ps.stem(w) + ' '

    # tokens = nltk.word_tokenize(corpus_no_stop_no_punch)
    # fdist = FreqDist(tokens)
    # print(i)
    # print(sorted(fdist.items(), key=lambda k: k[1], reverse=True)[:10])
    # print()