from sklearn.model_selection import train_test_split
import pandas as pd
from nltk import FreqDist
import nltk


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
            score += group_words[word] * row_words[word]
    denominator = sum(group_words.values())
    if denominator == 0:
        return -1
    score = score/denominator

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
    # print(row)
    # Taking only bullet points
    if not row['TITLE']:
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
    
    out_frame = pd.DataFrame()
    out_frame['PRODUCT_ID'] = test['PRODUCT_ID']
    out_frame['BROWSE_NODE_ID'] = test.apply(get_label_for_row, axis='index')
    out_frame.to_csv(outfile)


pd.set_option("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 200)

train = pd.read_csv(r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\train.csv', nrows=100000)
test = pd.read_csv(r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Processed\test.csv')
node_grp = train.groupby('BROWSE_NODE_ID', axis='index')

outfile = r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Predictions\pred_1.csv'
column_used = 'title'
top_n_group_words = 20

sorted_words = node_grp[column_used].apply(lambda series: sorted(FreqDist(nltk.word_tokenize(series.str.cat(sep=' '))).items(), key=lambda k: k[1], reverse=True)[:top_n_group_words])
group_word_freq = sorted_words.apply(lambda row: {key: val for key, val in row}).to_dict()

test_predictions(test, len(test))
