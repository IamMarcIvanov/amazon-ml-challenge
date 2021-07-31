import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer


# Downloads
nltk.download('stopwords')
nltk.download('wordnet')

def cleaning_function(data):
    global to_remove, stopWords, lem

    # print(f"{data}")
    if pd.isna(data):
        return None

    data = str(data)

    for_bullet_points = data.replace(',', ' ')

    # Removing puctuation and splitting string
    split_punc_removed = for_bullet_points.translate(str.maketrans('', '', to_remove)).split()

    # Removing stop_words and words with less length
    stop_words_removed = [word.lower() for word in split_punc_removed if len(word) > 3 and word.lower() not in stopWords]

    # Lemmatizing words and forming new data
    new_data = ' '.join([lem.lemmatize(word) for word in stop_words_removed])

    return new_data


ps = PorterStemmer()
lem = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))

infile = r'E:\Amazon ML Challenge\dataset52a7b21\dataset\train.csv'
outfile = r'E:\Amazon ML Challenge\dataset52a7b21\dataset\Processed\train2.csv'
step = 10000
to_remove = string.punctuation + string.digits

for i in range(10):
    print(f"i is {i}")
    df = pd.read_csv(infile, skiprows=step * i, nrows=step, names=['TITLE', 'DESCRIPTION', 'BULLET_POINTS', 'BRAND', 'BROWSE_NODE_ID'], escapechar="\\", quoting = 3, header=0)

    df['TITLE'] = df['TITLE'].apply(cleaning_function)
    df['DESCRIPTION'] = df['DESCRIPTION'].apply(cleaning_function)
    df['BULLET_POINTS'] = df['BULLET_POINTS'].apply(cleaning_function)

    df.to_csv(outfile, mode='w' if i==0 else 'a', header=True if i==0 else False, index=False)