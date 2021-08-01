import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import spacy
from sklearn.metrics import accuracy_score
import xgboost as xgb
import winsound

start = time.time()
nlp = spacy.load('en_core_web_md')
# pd.set_option("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 200)

data_path = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\train_lem_100k.csv'
#  keep_default_na=False, na_values=['(nan)', '()']
data = pd.read_csv(data_path, nrows=10000)

data['WORD_VEC'] = data.apply(lambda row: nlp(str(row['DESCRIPTION'])).vector, axis=1)
X_train, X_test, y_train, y_test = train_test_split(data['WORD_VEC'], data['BROWSE_NODE_ID'], test_size=0.3)
X_train = np.array(X_train.tolist())
X_test = np.array(X_test.tolist())
y_test = np.array(y_test.tolist())
y_train = np.array(y_train.tolist())

print('xgboost starting')
xgb_cl = xgb.XGBClassifier(use_label_encoder = False, eval_metric='mlogloss')
xgb_cl.fit(X_train, y_train)
preds = xgb_cl.predict(X_test)
print(accuracy_score(y_test, preds))


