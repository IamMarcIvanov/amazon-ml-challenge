import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
import time
import spacy
from transformers import pipeline

start = time.time()
nlp = spacy.load('en_core_web_md')
summarizer = pipeline("summarization")
pd.set_option("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 200)

datapath = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\train.csv'
data = pd.read_csv(datapath, nrows=100000)
train, test = train_test_split(data, test_size=0.2)
node_grp = train.groupby('BROWSE_NODE_ID', axis='index')
print(len(train['BROWSE_NODE_ID'].unique()))

desc_catted = node_grp['description'].apply(lambda series: series.str.cat(sep=' ')[:1020])
# print(desc_catted.head())

desc_summ = desc_catted.apply(lambda text: summarizer(text, max_length=150, min_length=1) if len(text.split()) > 150 else text)
desc_summ.to_csv(r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\desc_summary.csv')


print(time.time() - start)
