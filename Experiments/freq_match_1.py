from sklearn.model_selection import train_test_split
import pandas as pd
from nltk import FreqDist
import nltk
from rich.console import Console
from _collections import defaultdict
from collections import Counter
import time
import numpy as np

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
            # score += group_words[word] * row_words[word]
            score += row_words[word]

    # print(f"Group words is\n{group_words}")
    # denominator = sum(group_words.values())
    # if denominator == 0:
    #     return -1
    # score = score/denominator

    return score

def score_individual_column_all_groups(row_val, column_group_word_freq):

    score_dict = defaultdict(int)

    if pd.isna(row_val):
        # print(f"Could not predict for current row")
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
        # print(f"Using column {column}")
        all_scores[column] = score_individual_column_all_groups(row[column], group_word_freq_all[column])

    all_groups_set = set()
    for column in column_weights:
        all_groups_set = all_groups_set.union(set(all_scores[column].keys()))

    for group in all_groups_set:
        # print(f"Summing scores")
        score_sum = sum([all_scores[column][group]*weight for (column, weight) in column_weights.items()])
        if score_sum > max_score:
            predicted_group_label = group
            max_score = score_sum

    return predicted_group_label, max_score

    # Taking only bullet points
    if pd.isna(row[column_used]):
        # print(f"Could not predict for current row")
        return -1, -1

    # print(f"Row bullet point is {row['BULLET_POINTS']}")
    row_word_freq = FreqDist(nltk.word_tokenize(row[column_used]))

    for group, group_words in group_word_freq.items():
        score = score_individual_group(row_word_freq, group_words)
        if score > max_score:
            max_score = score
            predicted_group_label = group

    # print(f"Max_score is {max_score}. Group is {predicted_group_label}")
    # print(f"Actual group is {row['BROWSE_NODE_ID']}")
# ,max_score
    return predicted_group_label, max_score



def test_predictions(test_data, num_of_rows):
    # Counts accuracy as well
    global group_word_freq_all, column_weights

    rows_predicted_correctly = 0
    null_values = 0

    percent_increment = 0.5
    next_percent = percent_increment

    for i in range(num_of_rows):

        # print(f"Predicting for\n{test_data.iloc[i]}")
        # , score
        predicted_group_label, score= get_label_for_row(test_data.iloc[i])
        if predicted_group_label == test_data.iloc[i]['BROWSE_NODE_ID']:
            rows_predicted_correctly += 1
        # else:
        #     actual_group_label = test_data.iloc[i]['BROWSE_NODE_ID']
        #     row_word_freq = FreqDist(nltk.word_tokenize(test_data.iloc[i][column_used]))
        #     print(f"\nPREDICTED SCORE: {score}   ACTUAL SCORE: {score_individual_group(row_word_freq, group_word_freq[actual_group_label])}")
        #     print(f"predicted group: {predicted_group_label}  actual group: {actual_group_label}\n\ntest data : \n{test_data.iloc[i]}")
        #     console.print(f'predict word freq {group_word_freq[predicted_group_label]}\n\nactual word freq : {group_word_freq[actual_group_label]}')
        #     break

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

    # print(f"In group {group_number}")
    # print(f"Series is {series.values}")
    # print(f"All added {np.char.join(' ', series.values)}")
    # group_number += 1
    sorted_norm_freq =  Counter(' '.join(series.astype(str)).split()).most_common(top_n_group_words)
    # print(f"series is {series.}")
    # complete_text = series.str.cat(sep=' ')
    # freq_dict = FreqDist(nltk.word_tokenize(complete_text))
    # denom = len(series)
    # norm_freq_dict = {key: value/denom for (key, value) in freq_dict.items()}
    # sorted_norm_freq = sorted(freq_dict.items(), key=lambda kv: kv[1], reverse=True)[:top_n_group_words]

    return sorted_norm_freq

    # return pd.Series(' '.join(series.astype(str)).split()).value_counts()[:top_n_group_words].to_dict()



pd.set_option("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 200)
console = Console()

datapath = r'E:\Amazon ML Challenge\dataset52a7b21\dataset\Processed\stemmer_train.csv'
data = pd.read_csv(datapath, nrows=200000)
train, test = train_test_split(data, test_size=0.2)
node_grp = train.groupby('BROWSE_NODE_ID', axis='index')

column_weights = {'TITLE': 0.6, 'DESCRIPTION': 0.3, 'BRAND': 0.1}
top_n_group_words = 20

# test_series = pd.Series(['hi', 'hi Shar', 'Shar', 4])
# console.print(pd.Series(' '.join(test_series.astype(str)).split()).value_counts()[:top_n_group_words].to_dict())
group_number = 0

start = time.time()
sorted_words_all = {}
# group_word_freq_all = {}
for column in column_weights:
    print(f"Getting group frequencies for {column}")
    # group_number = 0
    # node_grp[column].aggregate(get_group_freq_dict)
    sorted_words_all[column] = node_grp[column].apply(get_group_freq_dict)
    # console.print("One group is\n",sorted_words_all[column][6])
    # sorted_words_all[column] = node_grp.apply(lambda series: pd.Series(' '.join(series).split()).value_counts()[:top_n_group_words])

end = time.time()
print(f"Time to get sorted_words={end-start}")
# lambda series: sorted(FreqDist(nltk.word_tokenize(series.str.cat(sep=' '))).items(), key=lambda k: k[1], reverse=True)[:top_n_group_words]

#FreqDist(nltk.word_tokenize(series.str.cat(sep=' '))).items(), key=lambda k: k[1], reverse=True)

start = time.time()
group_word_freq_all = {}
for column in column_weights:
    group_word_freq_all[column] = sorted_words_all[column].aggregate(lambda row: {key: val for key, val in row}).to_dict()
    # console.print("One group is\n", group_word_freq_all[column][6])
end = time.time()

print(f"Time to convert to dictionary={end-start}")
#console.print(list(group_word_freq.items())[:20])

test_predictions(test, 10000)


'''
with column -> TITLE
scoring -> weighted average  with lemmetization               38%
scoring -> direct addition with lemmetization                 42%
scoring -> multiply with weight and add with lemmetization    21%
scoring -> direct addition with porter stemmer                48%
scoring -> weighted average with porter stemmer               39-40%             
problems with scoring
1 -> 
2 ->
3 ->
problems with pre-processing
6 102*1 ->
7 2*5 -> 
6 -> []
[gismo oneplus back cover printed designer soft case oneplus design]
[0 0 20 0 19 18] -> [] 
5000    
40  50  60
3   4   2
'cover': 6359, 'back': 6012, 'case': 4656, 'printed': 3377,
'designer': 2511, 'hard': 2100, 'samsung': 1453, 'galaxy': 1357, 'mobile': 1245, 'soft': 1086, 'redmi': 921, 'vivo':
888, 'black': 812, 'design': 798, 'note': 720, 'oppo': 708, 'plus': 636, 'xiaomi': 609, 'iphone': 579, 'silicone': 560
flipflops / flipflop / flip flop -> ngram to battle this
{flip, lipf, ipfl, }
4 ->1
5 ->2
6 ->3
one plus/oneplus 
{'oneplus': 2, 'back': 2, 'cover': 2, 'case': 2, 'mobbysol®': 1, 'colour': 1, 'full': 1, 'protective':
1, 'plus': 1, 'five': 1, 'black': 1}, 
actual word freq : {'cover': 6359, 'back': 6012, 'case': 4656, 'printed': 3377,
'designer': 2511, 'hard': 2100, 'samsung': 1453, 'galaxy': 1357, 'mobile': 1245, 'soft': 1086, 'redmi': 921, 'vivo':
888, 'black': 812, 'design': 798, 'note': 720, 'oppo': 708, 'plus': 636, 'xiaomi': 609, 'iphone': 579, 'silicone': 560}
degre full bodi protect front back case cover ipaki style vivo gold
actual ->  2 
predicted -> 1
'''