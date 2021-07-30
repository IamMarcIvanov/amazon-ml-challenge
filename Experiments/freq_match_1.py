from sklearn.model_selection import train_test_split
import pandas as pd
from nltk import FreqDist
import nltk
from rich.console import Console

pd.set_option("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 200)
console = Console()

datapath = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\train.csv'
data = pd.read_csv(datapath, nrows=10000)
train, test = train_test_split(data, test_size=0.2)
node_grp = train.groupby('BROWSE_NODE_ID', axis='index')

sorted_words = node_grp['bulletpoints'].apply(lambda series: sorted(FreqDist(nltk.word_tokenize(series.str.cat(sep=' '))).items(), key=lambda k: k[1], reverse=True)[:10])
word_freq = sorted_words.apply(lambda row: {key: val for key, val in row}).to_dict()
console.print(list(word_freq.items())[:10])
