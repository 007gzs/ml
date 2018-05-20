# -*- coding: utf-8 -*-
from __future__ import absolute_import, unicode_literals

import csv
import argparse

import tensorflow as tf

import data
from data import SURVIVED

parser = argparse.ArgumentParser()
parser.add_argument('--skip_test', default=False, type=bool, help='skip test')
parser.add_argument('--batch_size', default=100, type=int, help='batch size')
parser.add_argument('--train_steps', default=1000, type=int,
                    help='number of training steps')


def main(argv):
    args = parser.parse_args(argv[1:])

    # Fetch the data
    (train_x, train_y), (test_x, test_y), test_real = data.load_data(skip_test=args.skip_test)

    # Feature columns describe how to use the input.
    my_feature_columns = []
    for key in train_x.keys():
        my_feature_columns.append(tf.feature_column.numeric_column(key=key))

    # Build 2 hidden layer DNN with 10, 10 units respectively.
    classifier = tf.estimator.DNNClassifier(
        feature_columns=my_feature_columns,
        # Two hidden layers of 10 nodes each.
        hidden_units=[10, 10, 10],
        # The model must choose between 2 classes.
        n_classes=2)

    # Train the Model.
    classifier.train(
        input_fn=lambda: data.train_input_fn(train_x, train_y,
                                             args.batch_size),
        steps=args.train_steps)
    if not args.skip_test:
        # Evaluate the model.
        eval_result = classifier.evaluate(
            input_fn=lambda: data.eval_input_fn(test_x, test_y,
                                                args.batch_size))

        print('\nTest set accuracy: {accuracy:0.3f}\n'.format(**eval_result))

    # Generate predictions from the model

    template = '\nPrediction is "{}" ({:.1f}%), id "{}"'
    predictions = classifier.predict(
        input_fn=lambda: data.eval_input_fn(test_real,
                                            labels=None,
                                            batch_size=args.batch_size))
    with open('submission.csv', 'w', newline='') as csvfile:
        csvwriter = csv.writer(csvfile)
        csvwriter.writerow(['PassengerId', 'Survived'])
        for pred_dict, realindex in zip(predictions, test_real.index):
            class_id = pred_dict['class_ids'][0]
            probability = pred_dict['probabilities'][class_id]
            csvwriter.writerow([realindex, class_id])
            print(template.format(SURVIVED[class_id],
                                  100 * probability, realindex))


if __name__ == "__main__":
    tf.logging.set_verbosity(tf.logging.INFO)
    tf.app.run(main)
