#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
disaster_tweets
Created by raj at 4:23 PM,  6/28/20
"""


import pandas as pd
from simpletransformers.classification import ClassificationModel

train = pd.read_csv('sample-data/train.csv')
test = pd.read_csv('sample-data/test.csv')


def split_to_train_test(df, label_column, train_frac=0.8):
    train_df, test_df = pd.DataFrame(), pd.DataFrame()
    labels = df[label_column].unique()
    for lbl in labels:
        print(type(lbl))
        lbl_df = df[df[label_column] == lbl]
        lbl_train_df = lbl_df.sample(frac=train_frac)
        lbl_test_df = lbl_df.drop(lbl_train_df.index)
        train_df = train_df.append(lbl_train_df)
        test_df = test_df.append(lbl_test_df)
    return train_df, test_df


train = train.rename(columns={'target': 'labels'})
train_df, eval_df = split_to_train_test(train, 'labels', 0.7)

train_df.reset_index(drop=True, inplace=True)
eval_df.reset_index(drop=True, inplace=True)
train_df = train_df[['text','labels']]
eval_df= eval_df[['text','labels']]

m = 'roberta-base'
m = 'outputs/best_model'
# Create a ClassificationModel
model = ClassificationModel('roberta', m,
                            args = {
                                'use_multiprocessing': False,
                                'reprocess_input_data': True,
                                'num_train_epochs': 45,
                                'learning_rate': 2e-05,
                                'evaluate_during_training': True,
                                'evaluate_during_training_verbose': True,
                                "evaluate_during_training_steps": 200,
                                "use_cached_eval_features": False,
                                "logging_steps": 200,
                                'use_early_stopping': True,
                                'early_stopping_patience': 7,
                                'weight_decay': 0.000001,
                                'do_lower_case': False,
                                "wandb_project": False,
                                'fp16_opt_level': 'O1',
                                'fp16': True,
                                'save_steps': 0,  # new from code inspect
                                "gradient_accumulation_steps": 1,
                                'save_eval_checkpoints': False,
                                'save_model_every_epoch': False
                            },
                            use_cuda = False,
) # You can set class weights by using the optional weight argument

# Train the model
# model.train_model(train_df, eval_df=eval_df, show_running_loss=True, verbose=True)

# Evaluate the model
# result, model_outputs, wrong_predictions = model.eval_model(eval_df)
# print(result)

pred, prob = model.predict(test['text'].tolist())
pred_df = pd.DataFrame(pred, columns=['target'])
pred['id'] = test['id']
pred.to_csv('submission.csv')
