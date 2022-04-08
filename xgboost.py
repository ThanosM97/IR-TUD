import argparse

import pandas as pd
from sklearn.metrics import classification_report
from pathlib import Path
from typing import Union

from xgboost import XGBClassifier


def test(dataset: Union[str, Path], checkpoint: Union[str, Path]) -> None:
    """ Test XBG model.

    This function evaluates on `dataset` a trained XGB classifier from
    `checkpoint` and prints the classification report.

    Arguments:
        - dataset (str, Path) :  Path to train dataset.
        - checkpoint (str, Path) : Path to checkpoint.
    """
    clf = XGBClassifier()
    clf.load_model(checkpoint)

    features = pd.read_csv(dataset, header=None, sep=" ").rename(
        columns={0: "target", 1: "qid", 2: "pid"})

    targets = features["target"].copy()
    features.drop(["target", "qid", "pid"], axis=1,  inplace=True)

    pred = clf.predict(features)

    print(classification_report(targets, pred))


def train(
        dataset: Union[str, Path],
        filename: Union[str, Path] = "XGB.model") -> None:
    """ Train XBG model.

    This function trains an XGB classifier using hist tree_method on a given
    `dataset` and saves the model on `filename`.

    Arguments:
        - dataset (str, Path) :  Path to train dataset.
        - filename (str, Path) : Filename for the saved model.
    """
    features = pd.read_csv(
        dataset, header=None, sep=" ").rename(
        columns={0: "target", 1: "qid", 2: "pid"})

    # Keep 1 negative for each positive example
    pos = features[features["target"] == 1]
    neg = features[features["target"] == 0].groupby("qid").head(1)

    features = pd.concat([pos, neg], axis=0).sample(frac=1)

    targets = features["target"].copy()
    features.drop(["target", "qid", "pid"], axis=1,  inplace=True)

    clf = XGBClassifier(tree_method='hist')
    clf.fit(features, targets)

    clf.save_model(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and test XGBoost classifier.")

    parser.add_argument(
        '--train', type=str, default=None, help="Path to train dataset.")
    parser.add_argument(
        '--save', type=str, default="XGB.model",
        help="Filename for the saved model.")

    parser.add_argument(
        '--test', type=str, default=None, help="Path to test dataset.")
    parser.add_argument(
        '--checkpoint', type=str, default=None,
        help="Path to checkpoint model.")

    args = parser.parse_args()

    if args.train is None and args.test is None:
        parser.error("Please specify either --train or --test path (or both)")

    if args.test is not None and args.checkpoint is None:
        parser.error(
            "Please specify path to a model's checkpoint [--checkpoint PATH]")

    if args.train is not None:
        train(args.train, args.save)

    if args.test is not None:
        test(args.test, args.checkpoint)
