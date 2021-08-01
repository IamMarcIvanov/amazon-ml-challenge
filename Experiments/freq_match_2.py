from sklearn.model_selection import train_test_split
import pandas as pd
from nltk import FreqDist
import nltk
from rich.console import Console
from collections import defaultdict
from collections import Counter
import time
import numpy as np
import cProfile
import pstats

# Download some data
# nltk.download('punkt')
def score_individual_group(row_words, group_words):
    """
    Calculate row score for a particular group
    :param row_words: Word frequency dict for row
    :param group_words: Group's word-frequency dict
    :return: score for (row, group)
    """
    score = 0
    for word in row_words:
        if word in group_words:
            score += row_words[word]

    return score

def score_individual_column_all_groups(row_val, column_group_word_freq):

    score_dict = defaultdict(int)

    if pd.isna(row_val):
        return score_dict

    row_word_freq = FreqDist(nltk.word_tokenize(row_val))

    for group, group_words in column_group_word_freq.items():
        score_dict[group] = score_individual_group(row_word_freq, group_words)

    return score_dict

def get_label_for_row(row):
    """
    Get score values for all groups and choose the highest value
    :param row:
    :return: BROWSE_NODE_ID for group with highest score
    """
    global group_word_freq_all, column_weights

    max_score = -1
    predicted_group_label = None

    all_scores = {column: defaultdict(int) for column in column_weights.keys()}

    for column in column_weights:
        all_scores[column] = score_individual_column_all_groups(row[column], group_word_freq_all[column])

    all_groups_set = set()
    for column in column_weights:
        all_groups_set = all_groups_set.union(set(all_scores[column].keys()))

    for group in all_groups_set:
        score_sum = sum([all_scores[column][group]*weight for (column, weight) in column_weights.items()])
        if score_sum > max_score:
            predicted_group_label = group
            max_score = score_sum

    return predicted_group_label, max_score

    # Taking only bullet points
    if pd.isna(row[column_used]):
        return -1, -1

    row_word_freq = FreqDist(nltk.word_tokenize(row[column_used]))

    for group, group_words in group_word_freq.items():
        score = score_individual_group(row_word_freq, group_words)
        if score > max_score:
            max_score = score
            predicted_group_label = group

    return predicted_group_label, max_score



def test_predictions(test_data, num_of_rows):
    # Counts accuracy as well
    global group_word_freq_all, column_weights

    rows_predicted_correctly = 0
    null_values = 0

    percent_increment = 0.5
    next_percent = percent_increment

    for i in range(num_of_rows):
        predicted_group_label, score= get_label_for_row(test_data.iloc[i])
        if predicted_group_label == test_data.iloc[i]['BROWSE_NODE_ID']:
            rows_predicted_correctly += 1

        if predicted_group_label == -1:
            null_values += 1

        # Print when additional 10% is done
        if ((i+1)*100/num_of_rows) >= next_percent:
            print(f"Percent done: {((i+1)*100/num_of_rows)}")
            print(f"Percent correct: {(rows_predicted_correctly/(i+1))*100}%")
            print(f"Null values till now: {null_values}, ({null_values*100/(i+1)}%)")
            next_percent += percent_increment

    print(f"Accuracy is {(rows_predicted_correctly/num_of_rows)*100}%")


def get_group_freq_dict(series):
    global top_n_group_words, group_number
    sorted_norm_freq =  Counter(' '.join(series.astype(str)).split()).most_common(top_n_group_words)

    return sorted_norm_freq


pd.set_option("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 200)
console = Console()

datapath = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\train_stem_100k.csv'
data = pd.read_csv(datapath, nrows=10000)
train, test = train_test_split(data, test_size=0.2)
node_grp = train.groupby('BROWSE_NODE_ID', axis='index')

column_weights = {'TITLE': 0.6, 'DESCRIPTION': 0.3, 'BRAND': 0.1}
top_n_group_words = 20

group_number = 0

start = time.time()
sorted_words_all = {}
for column in column_weights:
    print(f"Getting group frequencies for {column}")
    sorted_words_all[column] = node_grp[column].apply(get_group_freq_dict)

end = time.time()
print(f"Time to get sorted_words={end-start}")

start = time.time()
group_word_freq_all = {}
for column in column_weights:
    group_word_freq_all[column] = sorted_words_all[column].aggregate(lambda row: {key: val for key, val in row}).to_dict()
end = time.time()

print(f"Time to convert to dictionary={end-start}")

with cProfile.Profile() as pr:
    test_predictions(test, len(test))

stats = pstats.Stats(pr)
stats.sort_stats(pstats.SortKey.TIME)
stats.print_stats()