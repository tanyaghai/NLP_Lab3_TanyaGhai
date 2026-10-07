##
 # Harvey Mudd College, CS159
 # Swarthmore College, CS65
 # Copyright (c) 2018 Harvey Mudd College Computer Science Department, Claremont, CA
 # Copyright (c) 2018 Swarthmore College Computer Science Department, Swarthmore, PA
##

import argparse
import sys
from PCLDataReader import PCLLabels, PCLFeatures, PCLVocab

from sklearn.naive_bayes import MultinomialNB
from sklearn.dummy import DummyClassifier
import numpy as np

from sklearn.model_selection import cross_val_predict

class BinaryLabels(PCLLabels):
    """get the true/false condescension label for each example"""

    def _extract_label(self, example):
        """return the example's condescension attribute"""
        return example.get("condescension")


class CategoryLabels(PCLLabels):
    """get the topic category for each example."""

    def _extract_label(self, example):
        """return the example's category attribute"""
        return example.get("category")



class MyFeatures(PCLFeatures):

    def _extract_features(self, example):
        words = self.extract_text(example)
        features = []

        for word in words:
            if self.initial_vocab[word] is not None:
                features.append(word)

        return features

    def _get_feature_name(self, i):
        return self.vectorizer.get_feature_names_out()[i]

    def _get_num_features(self):
        return len(self.vectorizer.get_feature_names_out())

def do_experiment(args):

        vocab = PCLVocab(
            args.vocabulary,
            args.vocab_size,
            args.stop_words
        )


        features = MyFeatures(vocab)


        binary_labeler = BinaryLabels()
        category_labeler = CategoryLabels()


        model = MultinomialNB()


        args.data_file.seek(0)
        X, ids = features.process(args.data_file)

        args.data_file.seek(0)
        y = binary_labeler.process(args.data_file)

        args.data_file.seek(0)
        categories = category_labeler.process(args.data_file)

        y = np.array(y)
        categories = np.array(categories)
        ids = np.array(ids)


        if args.test_category is not None:
            category_id = category_labeler.labels[args.test_category]

            train_indices = np.where(categories != category_id)[0]
            test_indices = np.where(categories == category_id)[0]

            model.fit(X[train_indices], y[train_indices])
            probabilities = model.predict_proba(X[test_indices])

            class_ids = model.classes_
            output_ids = ids[test_indices]

        else:
            probabilities = cross_val_predict(
                model,
                X,
                y,
                cv=args.xvalidate,
                method="predict_proba"
            )

            class_ids = np.unique(y)
            output_ids = ids

        for example_id, row in zip(output_ids, probabilities):
            best_column = np.argmax(row)
            predicted_id = class_ids[best_column]
            predicted_label = binary_labeler[predicted_id]
            confidence = row[best_column]

            print(
                example_id,
                predicted_label,
                confidence,
                file=args.output_file
            )



    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("data_file", type=argparse.FileType('rb'), help="Data file containing labeled training instances")
    parser.add_argument("vocabulary", type=argparse.FileType('r'), help="File containing vocabulary words")
    parser.add_argument("-o", "--output_file", type=argparse.FileType('w'), default=sys.stdout, help="Write predictions to FILE", metavar="FILE")
    parser.add_argument("-v", "--vocab_size", type=int, metavar="N", help="Only count the top N words from the vocab file", default=None)
    parser.add_argument("-s", "--stop_words", type=int, metavar="N", help="Exclude the top N words as stop words", default=None)
    parser.add_argument("--train_size", type=int, metavar="N", help="Only train on the first N instances. N=0 means use all training instances.", default=None)

    eval_group = parser.add_mutually_exclusive_group(required=True)
    eval_group.add_argument("-t", "--test_category")
    eval_group.add_argument("-x", "--xvalidate", type=int)

    args = parser.parse_args()
    do_experiment(args)

    for fp in (args.data_file, args.vocabulary):
        fp.close()

    if args.output_file is not sys.stdout:
        args.output_file.close()