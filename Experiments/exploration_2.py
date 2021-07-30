import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer

lem = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))

infile = r'E:\Amazon ML Challenge\dataset52a7b21\dataset\train.csv'
outfile = r'E:\Amazon ML Challenge\dataset52a7b21\dataset\Processed\train.csv'
step = 10000
to_remove = string.punctuation + string.digits

for i in range(5):
    print(f"i is {i}")
    df = pd.read_csv(infile, skiprows=step * i, nrows=step, names=['TITLE', 'DESCRIPTION', 'BULLET_POINTS', 'BRAND', 'BROWSE_NODE_ID'], escapechar="\\", quoting = 3, header=0)

    df['TITLE'] = df['TITLE'].apply(lambda line: ' '.join([lem.lemmatize(w.lower()) for w in str(line).translate(str.maketrans('', '', to_remove)).split() if w.lower() not in stopWords and len(w) > 3]))
    df['DESCRIPTION'] = df['DESCRIPTION'].apply(lambda line: ' '.join([lem.lemmatize(w.lower()) for w in str(line).translate(str.maketrans('', '', to_remove)).split() if w.lower() not in stopWords and len(w) > 3]))
    df['BULLET_POINTS'] = df['BULLET_POINTS'].apply(lambda line: ' '.join([lem.lemmatize(w.lower()) for w in str(line).strip('[]').replace(',', ' ').translate(str.maketrans('', '', to_remove)).split() if w.lower() not in stopWords and len(w) > 3]))

    df.to_csv(outfile, mode='w' if i==0 else 'a', header=True if i==0 else False, index=False)