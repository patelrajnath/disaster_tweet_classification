#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
disaster_tweets
Created by raj at 4:23 PM,  6/28/20
"""


import pandas as pd
from simpletransformers.classification import ClassificationModel

train = pd.read_csv('sample-data/train.csv')
# test = pd.read_csv('sample-data/test.csv')


def split_to_train_test(df, label_column, train_frac=0.8):
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    labels = df[label_column].unique()
    for lbl in labels:
        lbl_df = df[df[label_column] == lbl]
        lbl_train_df = lbl_df.sample(frac=train_frac)
        lbl_test_df = lbl_df.drop(lbl_train_df.index)
        train_df = train_df.append(lbl_train_df)
        test_df = test_df.append(lbl_test_df)
    return train_df, test_df


train = train.rename(columns={'target': 'labels'})
train_df, eval_df = split_to_train_test(train, 'labels', 0.7)

# Create a ClassificationModel
model = ClassificationModel('roberta',
                            'roberta-base',
                            args = {
                                'fp16' : False,
                                'num_train_epochs': 1,
                                'evaluate_during_training': True,
                                'evaluate_during_training_steps': 200
                            },
                            use_cuda = False,
) # You can set class weights by using the optional weight argument

# Train the model
model.train_model(train_df, eval_df=eval_df, show_running_loss=True, verbose=True)

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(eval_df)

print(result)
