import pandas as pd
import nltk
from nltk.corpus import stopwords
import string
from nltk.stem import WordNetLemmatizer
from nltk.stem import PorterStemmer
import math


# Downloads
# nltk.download('stopwords')
# nltk.download('wordnet')

def cleaning_function(data):
    if pd.isna(data):
        return None

    data = str(data)

    for_bullet_points = data.replace(',', ' ')

    # Removing puctuation and splitting string
    split_punc_removed = for_bullet_points.translate(str.maketrans('', '', to_remove)).split()

    # Removing stop_words and words with less length
    stop_words_removed = [word.lower() for word in split_punc_removed if len(word) > 3 and word.lower() not in stopWords]

    # Lemmatizing words and forming new data
    #new_data = ' '.join([lem.lemmatize(word) for word in stop_words_removed])

    # TRIAL -> with porter stemmer
    new_data = ' '.join([ps.stem(word) for word in stop_words_removed])    

    return new_data


ps = PorterStemmer()
lem = WordNetLemmatizer()
stopWords = set(stopwords.words('english'))

infile = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\test.csv'
outfile = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\test.csv'
step = 10000
to_remove = string.punctuation + string.digits
for i in range(math.ceil(110775 / step)):
    print(f"i is {i}")
    df = pd.read_csv(infile, skiprows=step * i, nrows=step, names=['PRODUCT_ID', 'TITLE', 'DESCRIPTION', 'BULLET_POINTS', 'BRAND'], escapechar="\\", quoting = 3, header=0)

    df['TITLE'] = df['TITLE'].apply(cleaning_function)
    df['DESCRIPTION'] = df['DESCRIPTION'].apply(cleaning_function)
    df['BULLET_POINTS'] = df['BULLET_POINTS'].apply(cleaning_function)

    df.to_csv(outfile, mode='w' if i==0 else 'a', header=True if i==0 else False, index=False)



"""
train2 -> lemmetization
train3 -> porter stemmer

"""