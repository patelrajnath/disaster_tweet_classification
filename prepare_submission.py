import pandas as pd

test = pd.read_csv('sample-data/test.csv')
pred = pd.read_csv('submission.csv')

pred['id'] = test['id']
pred.to_csv('submission_final.csv', index=False)
