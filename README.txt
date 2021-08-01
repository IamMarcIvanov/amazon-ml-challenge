Approach 

Preprocess text - 
Remove stop words, stemming, punctuation and numbers

"Training phase"-
For each value in the BROWSE_NODE_ID(group) column, create a dictionary with the 20 most frequent words for each column in the training data. 

While predicting-
Find the frequencies of all words in sample in the test data
Each sample is assigned a score that tells how likely it is to belong to a group (BROWSE_NODE_ID)
The predicted value of BROWSE_NODE_ID is the one that has the maximum score for that particular sample.

Scoring system - 
The score is the sum of the frequency of the words in the sample that also occur in the group for which the score is being calculated.

Tools used:
1. pandas
2. numpy
3. cProfile
4. pstats
5. sklearn


