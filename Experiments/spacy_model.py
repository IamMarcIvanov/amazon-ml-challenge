
from sklearn.model_selection import train_test_split
import pandas as pd
import numpy as np
from nltk import FreqDist
import nltk
from rich.console import Console
import spacy
import timeit



# Load the spacy model that you have installed
nlp = spacy.load('en_core_web_md')

def score_individual_group(row_words, group_words):
    """
    Calculate row score for a particular group
    :param row_words: Word frequency dict for row
    :param group_words: Group's word-frequency dict
    :return: score for (row, group)
    """

    global vector_word_dict, freq_word_dict
    score = 0
    
    for word in row_words:
        word1 = nlp(word)
        for grp_word in group_words:
            word2 = vector_word_dict[grp_word]
            sim_score = word1.similarity(word2)
            score += (sim_score * group_words[grp_word] * row_words[word])/freq_word_dict[grp_word]
            # score += row_words[word]

    # print(f"Group words is\n{group_words}")
    # denominator = sum(group_words.values())
    # if denominator == 0:
    #     return -1
    # score = score/denominator

    return score

def get_label_for_row(row):
    """
    Get score values for all groups and choose the highest value
    :param row:
    :return: BROWSE_NODE_ID for group with highest score
    """
    global group_word_freq, column_used

    max_score = -1
    predicted_group_label = None

    # Taking only bullet points
    if pd.isna(row[column_used]):
        # print(f"Could not predict for current row")
        return -1

    # print(f"Row bullet point is {row['BULLET_POINTS']}")
    row_word_freq = FreqDist(nltk.word_tokenize(row[column_used]))

    for group, group_words in group_word_freq.items():
        score = score_individual_group(row_word_freq, group_words)
        if score > max_score:
            max_score = score
            predicted_group_label = group

    # print(f"Max_score is {max_score}. Group is {predicted_group_label}")
    # print(f"Actual group is {row['BROWSE_NODE_ID']}")

    return predicted_group_label

def test_predictions(test_data, num_of_rows):
    # Counts accuracy as well
    global group_word_freq, column_used

    rows_predicted_correctly = 0
    null_values = 0

    percent_increment = 2.5
    next_percent = percent_increment

    for i in range(num_of_rows):

        # print(f"Predicting for\n{test_data.iloc[i]}")
        predicted_group_label= get_label_for_row(test_data.iloc[i])
        if predicted_group_label == test_data.iloc[i]['BROWSE_NODE_ID']:
            rows_predicted_correctly += 1
        else:
            actual_group_label = test_data.iloc[i]['BROWSE_NODE_ID']
            row_word_freq = FreqDist(nltk.word_tokenize(test_data.iloc[i][column_used]))
            print(f"predicted group {predicted_group_label}\nactual group {actual_group_label}\ntest data : {test_data.iloc[i]}")
            console.print(f'predict word freq {group_word_freq[predicted_group_label]}\nactual word freq : {group_word_freq[actual_group_label]}')
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


console = Console()

datapath = r'/mnt/d/amazon-ml/dataset/Processed/train3.csv'
data = pd.read_csv(datapath, nrows=100000)
train, test = train_test_split(data, test_size=0.2)
node_grp = train.groupby('BROWSE_NODE_ID', axis='index')

column_used = 'TITLE'
top_n_group_words = 20

sorted_words = node_grp[column_used].apply(lambda series: sorted(FreqDist(nltk.word_tokenize(series.str.cat(sep=' '))).items(), key=lambda k: k[1], reverse=True)[:top_n_group_words])

#FreqDist(nltk.word_tokenize(series.str.cat(sep=' '))).items(), key=lambda k: k[1], reverse=True)


group_word_freq = sorted_words.apply(lambda row: {key: val for key, val in row}).to_dict()

vector_word_dict = {}
freq_word_dict = {}
lst = list(group_word_freq.items())

start = timeit.timeit()

for grp in lst:
    top_words=grp[1]
    for word, freq in top_words.items():
        if word not in vector_word_dict:
            vector_word_dict[word] = nlp(word)
            freq_word_dict[word] = freq
        else:
            freq_word_dict[word] += freq 
end = timeit.timeit()
print(f"DONE : vector and freq list for unique words made in {end-start}")
test_predictions(test, 10000)   
            





"""

word- > vector 
(word)

file row difference

file -> folder 0.8, file , binder, rose, tulip 0.1 ... n unique words

blow -> hit 0.2 run 

1 2 3 4 5 6 7 8
1 0 3 4 1 1 2 6  -> 18
1/18 0/18 3/18

1/18 * 0.8 + 0.2 *

Brand 
9 -> oneplus, mi, 
10 -> 

Title -> direct addition with porter stemmer 2 min
Description -> 

sentence -> word vector

how to formulate the word vector for a group -> text summarizer


"""