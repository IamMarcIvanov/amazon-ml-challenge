import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))

infile = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\train.csv'
outfile = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\train.csv'
step = 50000
to_remove = string.punctuation + string.digits
with open(outfile, 'w'):
    pass
for i in range(2):
    df = pd.read_csv(infile, skiprows=step * i, nrows=step, names = ['TITLE', 'DESCRIPTION', 'BULLET_POINTS', 'BRAND', 'BROWSE_NODE_ID'], escapechar="\\", quoting = 3)
    df['TITLE'] = df['TITLE'].apply(lambda line: ' '.join([lem.lemmatize(w.lower()) for w in str(line).translate(str.maketrans('', '', to_remove)).split() if w.lower() not in stopWords and len(w) > 3]))
    df['DESCRIPTION'] = df['DESCRIPTION'].apply(lambda line: ' '.join([lem.lemmatize(w.lower()) for w in str(line).translate(str.maketrans('', '', to_remove)).split() if w.lower() not in stopWords and len(w) > 3]))
    df['BULLET_POINTS'] = df['BULLET_POINTS'].apply(lambda line: ' '.join([lem.lemmatize(w.lower()) for w in str(line).strip('[]').replace(',', ' ').translate(str.maketrans('', '', to_remove)).split() if w.lower() not in stopWords and len(w) > 3]))
    df.to_csv(outfile, mode='a', header=False, index=False)