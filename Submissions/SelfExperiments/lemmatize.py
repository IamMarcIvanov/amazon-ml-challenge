import dask.dataframe as dd
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import traceback
import winsound
import cProfile
import pstats
import time
import string
from datetime import datetime

stopWords = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()

class Lem:
    def __init__(self, in_file_path):
        self.train = dd.read_csv(in_file_path, escapechar="\\", quoting = 3)
        self.to_remove = string.punctuation + string.digits
    
    def write_lemmatized(self, out_file_path):
        self.train[['TITLE', 'DESCRIPTION', 'BROWSE_NODE_ID']] = self.train[['TITLE', 'DESCRIPTION', 'BROWSE_NODE_ID']].applymap(self.preprocess)
        self.train.to_csv(out_file_path, single_file=True, compute=True)
    
    def preprocess(self, s):
        unpunctuated = str(s).replace(',', ' ').translate(str.maketrans('', '', self.to_remove)).split()
        return ' '.join([lemmatizer.lemmatize(word.lower()) for word in unpunctuated if len(word) > 3 and word not in stopWords])

def main():
    with cProfile.Profile() as pr:
        train_file_path = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\train.csv'
        lemmatized_file_path = r'D:\Svalbard\Data\AmazonMLChallengeData\SelfExperimentsData\train_all_lemm.csv'
        obj = Lem(train_file_path)
        obj.write_lemmatized(lemmatized_file_path)
    
    
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    suffix = str(datetime.now()).replace(':', '-').replace(' ', '_')
    stats.dump_stats(filename=r'D:\LibraryOfBabel\Projects\AmazonMLChallenge\Experiments\SelfExperiments\Performance\lemmatize' + suffix + '.prof')


if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
        winsound.Beep(1000, 500)

