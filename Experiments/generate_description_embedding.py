import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import spacy
from sklearn.metrics import accuracy_score
import winsound
import cProfile
import pstats
import time

def get_prediction(row, nlp, summ):
    if not row['DESCRIPTION']:
        return -1
    desc = nlp(str(row['DESCRIPTION']))
    summ['SIM_SCORE'] = summ.apply(lambda inner: np.dot(nlp(str(inner['DESCRIPTION'])).vector, desc.vector) / (np.linalg.norm(desc.vector) * np.linalg.norm(nlp(str(inner['DESCRIPTION'])).vector)) if np.linalg.norm(nlp(str(inner['DESCRIPTION'])).vector) > 0 and np.linalg.norm(desc.vector) > 0 else -1, axis=1)
    print(summ['BROWSE_NODE_ID'].iloc[summ['SIM_SCORE'].idxmax()])
    return summ['BROWSE_NODE_ID'].iloc[summ['SIM_SCORE'].idxmax()]

def get_data():
    summarised_file = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\summary_lemma.csv'
    # data_path = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\train_lem_100k.csv'
    # data = pd.read_csv(data_path, nrows=10)
    summ = pd.read_csv(summarised_file)
    return summ# , data

def main():
    with cProfile.Profile() as pr:
        start = time.time()
        nlp = spacy.load('en_core_web_md')
        pd.set_option("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 200)
        summ = get_data()
        
        # train, test = train_test_split(data, test_size=0.3)
        # pred = test.apply(lambda row: get_prediction(row, nlp, summ), axis=1)
        # print('accuracy', accuracy_score(test['BROWSE_NODE_ID'], pred))
        summ['DESCRIPTION_VECTORS'] = summ.apply(lambda row: nlp(str(row['DESCRIPTION'])).vector, axis=1)
        summ.to_csv(r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\summary_with_embedding.csv')
        print(time.time() - start)
    
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.print_stats()
    stats.dump_stats(filename=r'D:\LibraryOfBabel\Projects\AmazonMLChallenge\Experiments\Performance\gen_desc_vec_perf2.prof')
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        winsound.Beep(1000, 1000)
