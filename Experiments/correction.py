import pandas as pd
import random

test = pd.DataFrame()
test['PRODUCT_ID'] = list(range(1, 110776))
test['BROWSE_NODE_ID'] = 1046# random.choices(list(range(1, 10000)), k=len(test))
test.to_csv(r'D:\Svalbard\Data\AmazonMLChallengeData\dataset\Predictions\Pred_9')