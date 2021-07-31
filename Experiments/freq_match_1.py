from sklearn.model_selection import train_test_split
import pandas as pd
from nltk import FreqDist
import nltk
from rich.console import Console

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
            # score += row_words[word]

    # print(f"Group words is\n{group_words}")
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
        # else:
        #     actual_group_label = test_data.iloc[i]['BROWSE_NODE_ID']
        #     row_word_freq = FreqDist(nltk.word_tokenize(test_data.iloc[i][column_used]))
        #     print(f"\nPREDICTED SCORE : {score}   ACTUAL SCORE {score_individual_group(row_word_freq, group_word_freq[actual_group_label])}")
        #     print(f"predicted group {predicted_group_label}\nactual group {actual_group_label}\ntest data : {test_data.iloc[i]}")
        #     console.print(f'predict word freq {group_word_freq[predicted_group_label]}\nactual word freq : {group_word_freq[actual_group_label]}')
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


pd.set_option("display.max_rows", None, "display.max_columns", None, "display.max_colwidth", 200)
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