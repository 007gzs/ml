# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import pandas as pd
import re
import tensorflow as tf


SURVIVED = ['Died', 'Live']


def get_title(name):
    title_search = re.search('([A-Za-z]+)\.', name)
    if title_search:
        return title_search.group(1)
    return ""


def get_index(x, l):
    ret = -1
    for i, d in enumerate(l):
        if x > d:
            ret = i
        else:
            return ret
    return ret


def proc_data(df):
    title_mapping = {"Mr": 1, "Miss": 2, "Mrs": 3, "Master": 4, "Dr": 5, "Rev": 6}
    df['Title'] = df['Name'].apply(get_title).map(lambda x: title_mapping.get(x, 0))
    df['Cabin'] = df['Cabin'].fillna(0).apply(lambda x: 0 if x == 0 else 1)
    df['Embarked'] = df['Embarked'].fillna(df['Embarked'].dropna().mode()[0])
    maps = {
        'Sex': {'male': 1, 'female': 2},
        'Embarked': {'S': 1, 'C': 2, 'Q': 3}
    }
    for k, v in maps.items():
        df[k] = df[k].map(v)
    df['Age'] = df['Age'].fillna(-1)
    df['Age'] = df['Age'].map(lambda x: get_index(x, [0, 12, 18, 30, 40, 70]))
    df['Fare'] = df['Fare'].map(lambda x: get_index(x, [0, 10, 20, 30, 50, 100]))
    df = df.drop(['Ticket', 'Name'], axis=1)
    return df


def load_data(y_name='Survived', skip_test=False):
    train_path = 'train.csv'
    test_path = 'test.csv'

    train = proc_data(pd.read_csv(train_path, index_col="PassengerId"))
    size = len(train)
    test_size = 0 if skip_test else max(int(size * 0.2), 1)
    test = train.tail(test_size)
    train = train.head(size - test_size)
    test_real = proc_data(pd.read_csv(test_path, index_col="PassengerId"))

    train_x, train_y = train, train.pop(y_name)
    # if y_name not in test:
    #     test[y_name] = None
    test_x, test_y = test, test.pop(y_name)

    return (train_x, train_y), (test_x, test_y), test_real


def train_input_fn(features, labels, batch_size):
    """An input function for training"""
    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    # Shuffle, repeat, and batch the examples.
    dataset = dataset.shuffle(1000).repeat().batch(batch_size)

    # Return the dataset.
    return dataset


def eval_input_fn(features, labels, batch_size):
    """An input function for evaluation or prediction"""
    features = dict(features)
    if labels is None:
        # No labels, use only features.
        inputs = features
    else:
        inputs = (features, labels)

    # Convert the inputs to a Dataset.
    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    # Batch the examples
    assert batch_size is not None, "batch_size must not be None"
    dataset = dataset.batch(batch_size)

    # Return the dataset.
    return dataset


# # The remainder of this file contains a simple example of a csv parser,
# #     implemented using the `Dataset` class.
#
# # `tf.parse_csv` sets the types of the outputs to match the examples given in
# #     the `record_defaults` argument.
# CSV_TYPES = [[0.0], [0.0], [0.0], [0.0], [0]]
#
# def _parse_line(line):
#     # Decode the line into its fields
#     fields = tf.decode_csv(line, record_defaults=CSV_TYPES)
#
#     # Pack the result into a dictionary
#     features = dict(zip(CSV_COLUMN_NAMES, fields))
#
#     # Separate the label from the features
#     label = features.pop('Species')
#
#     return features, label
#
#
# def csv_input_fn(csv_path, batch_size):
#     # Create a dataset containing the text lines.
#     dataset = tf.data.TextLineDataset(csv_path).skip(1)
#
#     # Parse each line.
#     dataset = dataset.map(_parse_line)
#
#     # Shuffle, repeat, and batch the examples.
#     dataset = dataset.shuffle(1000).repeat().batch(batch_size)
#
#     # Return the dataset.
#     return dataset
