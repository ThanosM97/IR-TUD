"""This module performs a qid-aware train/dev split"""
import argparse

import numpy as np
import pandas as pd


def main(dataset: str) -> None:
    name = dataset.split("/")[-1].split(".")[0]
    # Read the training features
    train = pd.read_csv(
        dataset, sep=" ", header=None)

    # Sample 10% of the qids for the development set
    val_keys = np.random.choice(
        train[1].unique(),
        int(0.1 * len(train[1].unique())))
    val = train[train[1].isin(val_keys)]

    # Keep the rest 90% for the training set
    new_train = train[~train[1].isin(val_keys)]

    # Save the corresponding files
    new_train.to_csv(f"{name}_train_90.txt", sep=" ",
                     index=False, header=None)
    val.to_csv(f"{name}_dev_10.txt", sep=" ", index=False, header=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Split dataset to train/dev sets.")

    parser.add_argument(
        '-d', type=str, required=True, help="Path to dataset.")

    args = parser.parse_args()

    main(args.d)
