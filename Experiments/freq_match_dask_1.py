# from sklearn.model_selection import train_test_split
import pandas as pd
# from rich.console import Console
import traceback
# from sklearn.metrics import accuracy_score
from collections import defaultdict
from collections import Counter
import time
# import numpy as np
import cProfile
import pstats
# import dask.array as da
import dask.dataframe as dd
from dask_ml.metrics import accuracy_score
from dask_ml.model_selection import train_test_split
import winsound

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

    row_word_freq = Counter(row_val.split())
    print(column_group_word_freq.head())
    for group, group_words in column_group_word_freq.items():
        score_dict[group] = score_individual_group(row_word_freq, group_words)

    return score_dict

def get_label_for_row(row, group_word_freq_all, column_weights):
    """
    Get score values for all groups and choose the highest value
    :param row:
    :return: BROWSE_NODE_ID for group with highest score
    """

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

    return predicted_group_label

def test_predictions(test_data, num_of_rows, group_word_freq_all, column_weights):    
    predicted_group_label = test_data.apply(lambda row: get_label_for_row(row, group_word_freq_all, column_weights), meta='int', axis=1)
    # print(sum(da.where(test_data['BROWSE_NODE_ID'].fillna(value=1045) == predicted_group_label.fillna(value=1045))))
    print(f"Accuracy is {accuracy_score(test_data['BROWSE_NODE_ID'].fillna(value=1045), predicted_group_label.fillna(value=1045)) * 100} %")

def main():
    start = time.time()
    
    datapath = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\train_stem_100k.csv'
    data = dd.read_csv(datapath)
    train, test = train_test_split(data, test_size=0.2, shuffle=True)
    node_grp = train.groupby('BROWSE_NODE_ID')
    column_weights = {'TITLE': 0.6, 'DESCRIPTION': 0.3, 'BRAND': 0.1}
    top_n_group_words = 20
    group_number = 0
    
    group_word_freq_all = {column: node_grp[column].apply(lambda row: dict(Counter(' '.join(row.astype(str)).split()).most_common(top_n_group_words)), meta='dict') for column in column_weights}
    # group_word_freq_all = {column: sorted_words_all[column].apply(lambda row: {key: val for key, val in row}, meta='dict').to_dict() for column in column_weights}
    
    with cProfile.Profile() as pr:
        test_predictions(test, len(test), group_word_freq_all, column_weights)
    
    stats = pstats.Stats(pr)
    stats.sort_stats(pstats.SortKey.TIME)
    stats.dump_stats(filename=r'D:\LibraryOfBabel\Projects\AmazonMLChallenge\Experiments\Performance\freq_match_2_perf6.prof')
    
if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(e)
        traceback.print_exc()
        winsound.Beep(1000, 1000)