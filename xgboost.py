"""This module implements training of the XGBoost classifier and rakking."""
import argparse

import pandas as pd
from sklearn.metrics import classification_report
from pathlib import Path
from typing import Union

from xgboost import XGBClassifier


def test(dataset: Union[str, Path], checkpoint: Union[str, Path]) -> None:
    """ Test XBG model.

    This function generates a ranking for the `dataset` using a trained XGB 
    classifier from `checkpoint`.

    Arguments:
        - dataset (str, Path) :  Path to train dataset.
        - checkpoint (str, Path) : Path to checkpoint.
    """
    clf = XGBClassifier()
    clf.load_model(checkpoint)

    features = pd.read_csv(dataset, header=None, sep=" ").rename(
        columns={0: "target", 1: "qid", 2: "pid"})

    targets = features["target"].copy()
    ids = features[["qid", "pid"]].copy()
    features.drop(["target", "qid", "pid"], axis=1,  inplace=True)

    pred = clf.predict_proba(features)
    pred = pd.DataFrame(pred[:, 1], columns=["pred"])

    ranking = pd.concat([ids, pred], axis=1)

    # Sort per query-group of passages
    ranking = ranking.groupby('qid').apply(lambda x: x.sort_values(
        'pred', ascending=False)).add_suffix('t').reset_index().drop([
            "qid", "level_1"],
        axis=1)

    # There are 1000 passages per query
    ranking["predt"] = ranking.index % 1000 + 1

    ranking.to_csv("XGB_ranking.txt", header=None, sep="\t", index=False)


def train(
        dataset: Union[str, Path],
        val_dataset: Union[str, Path],
        filename: Union[str, Path] = "XGB.model") -> None:
    """ Train XBG model.

    This function trains an XGB classifier using hist tree_method on a given
    `dataset` and saves the model on `filename`.

    Arguments:
        - dataset (str, Path) :  Path to train dataset.
        - val_dataset (str, Path) :  Path to validation dataset.
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

    # Train
    clf = XGBClassifier(tree_method='hist')
    clf.fit(features, targets)

    # Validate
    val_features = pd.read_csv(
        val_dataset, header=None, sep=" ").rename(
        columns={0: "target", 1: "qid", 2: "pid"})
    val_targets = val_features["target"].copy()
    val_features.drop(["target", "qid", "pid"], axis=1,  inplace=True)

    pred = clf.predict(features)
    print(classification_report(val_targets, pred))

    clf.save_model(filename)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train and test XGBoost classifier.")

    parser.add_argument(
        '--train', type=str, default=None, help="Path to train dataset.")
    parser.add_argument(
        '--val', type=str, default=None, help="Path to validation dataset.")
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
        train(args.train, args.val, args.save)

    if args.test is not None:
        test(args.test, args.checkpoint)
