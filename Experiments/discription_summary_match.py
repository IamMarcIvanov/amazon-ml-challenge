import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import spacy
from sklearn.metrics import accuracy_score
import winsound

start = time.time()
nlp = spacy.load('en_core_web_md')
# pd.set_option("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 200)

summarised_file = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\summary_lemma.csv'
data_path = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\train_lem_100k.csv'
data = pd.read_csv(data_path, nrows=1000)
summ = pd.read_csv(summarised_file)

train, test = train_test_split(data, test_size=0.3)

i = 0
def get_prediction(row):
    global i
    i += 1
    print(i)
    print(time.time() - start)
    if not row['DESCRIPTION']:
        return -1
    desc = nlp(str(row['DESCRIPTION']))
    # summ['SIM_SCORE'] = summ.apply(lambda inner: nlp(str(inner['DESCRIPTION'])).similarity(desc) if inner['DESCRIPTION'] else -1, axis=1)
    # (np.dot(nlp(str(inner['DESCRIPTION'])).vector, desc.vector) / (np.linalg.norm(desc.vector) * np.linalg.norm(nlp(str(inner['DESCRIPTION'])).vector))
    summ['SIM_SCORE'] = summ.apply(lambda inner: np.dot(nlp(str(inner['DESCRIPTION'])).vector, desc.vector) / (np.linalg.norm(desc.vector) * np.linalg.norm(nlp(str(inner['DESCRIPTION'])).vector)) if np.linalg.norm(nlp(str(inner['DESCRIPTION'])).vector) > 0 and np.linalg.norm(desc.vector) > 0 else -1, axis=1)
    print(summ['BROWSE_NODE_ID'].iloc[summ['SIM_SCORE'].idxmax()])
    return summ['BROWSE_NODE_ID'].iloc[summ['SIM_SCORE'].idxmax()]

pred = test.apply(get_prediction, axis=1)
print('accuracy', accuracy_score(test['BROWSE_NODE_ID'], pred))

print(time.time() - start)
winsound.Beep(1000, 1000)
