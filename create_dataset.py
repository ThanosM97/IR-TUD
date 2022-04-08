"""This module generates the train and test datasets."""
import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from gensim.models import KeyedVectors
from gensim.scripts.glove2word2vec import glove2word2vec
from numpy.linalg import norm
from pyserini.index import IndexReader
from tqdm import tqdm

from .vectorizer import Word2VecVectorizer


def matching_terms(query: str, passage: str, index_reader: IndexReader) -> int:
    """Return the common terms of the `query`-`passage` pair."""
    query = index_reader.analyze(query)
    return len(set(query) & set(passage))


def cosine(x, y):
    """Calculate cosine similarity."""
    return np.dot(x, y)/(norm(x)*norm(y))


def main(args):
    # Read queries
    queries = pd.read_csv(args.queries, header=None,
                          sep="\t", names=["qid", "query"])

    if args.split == "train":
        df = pd.read_csv(args.dataset)
    else:
        df = pd.read_csv(args.dataset, sep="\t", header=None,
                         names=["qid", "pid", "rank"])
        df = pd.merge(df, queries, how="left", on="qid")

    # Load pyserini's IndexReader
    index_reader = IndexReader(args.indexes)

    if args.glove is not None:
        path = Path(args.glove)
        word2vec_file = path.with_suffix('.word2vec')

        if not word2vec_file.is_file():
            print("Converting Glove's weights to word2vec format..")
            glove2word2vec(path, word2vec_file)

        # Load model
        model = KeyedVectors.load_word2vec_format(
            word2vec_file, binary=False)

        # Create vectorizer
        vectorizer = Word2VecVectorizer(model)

    elif args.fasttext is not None:
        path = Path(args.feasttext)
        # Load model
        model = KeyedVectors.load_word2vec_format(
            path, binary=False)

        # Create vectorizer
        vectorizer = Word2VecVectorizer(model)
    else:
        vectorizer = None

    if args.split == "train":
        # Split query-positive-negative triplets
        pos = df.drop("neg_id", axis=1).drop_duplicates().reset_index(
            drop=True).rename(columns={"pos_id": "pid"})
        neg = df.drop("pos_id", axis=1).drop_duplicates().reset_index(
            drop=True).rename(columns={"neg_id": "pid"})

        # Add relevance labels
        pos["relevance"] = 1
        neg["relevance"] = 0

        merged_pos = pd.merge(pos, queries, how="inner", on="qid")
        merged_neg = pd.merge(neg, queries, how="inner", on="qid")

        # Create final set of examples
        df = pd.concat([merged_pos, merged_neg]).reset_index(drop=True)

    # Baseline L2R features
    df["score"] = 0.0
    df["tfidf"] = 0.0
    df["qBM25"] = 0.0
    df["nterms"] = 0
    df["qlength"] = 0
    df["plength"] = 0

    # Cosine and Euclidean similarities for Query-Passage pairs
    if vectorizer is not None:
        df["cosine"] = 0.0
        df["euclidean"] = 0.0

    for i in tqdm(df.index):
        passage = index_reader.get_document_vector(str(df.at[i, "pid"])).keys()

        # Query-Passage score
        df.at[i, "score"] = index_reader.compute_query_document_score(
            str(df.at[i, "pid"]), df.at[i, "query"])

        # Query-Passage matching terms
        df.at[i, "nterms"] = matching_terms(
            df.at[i, "query"], passage, index_reader)

        # Passage tfidf
        df.at[i, "tfidf"] = np.mean(np.multiply(
            list(
                index_reader.get_document_vector(
                    str(df.at[i, "pid"])).values()),
            [np.log(
                index_reader.stats()["documents"] /
                (1 +
                    (index_reader.get_term_counts(
                        term, analyzer=None))[0]))
                for term in index_reader.get_document_vector(
                str(df.at[i, "pid"])).keys()]))

        # Passage length
        df.at[i, "plength"] = len(passage)

        # Query BM25 score
        df.at[i, "qBM25"] = np.mean(
            [index_reader.compute_bm25_term_weight(str(df.at[i, "pid"]), word,
                                                   analyzer=None)
             for word in index_reader.analyze(df.at[i, "query"])])

        # Query length
        df.at[i, "qlength"] = len(df.at[i, "query"].split(" "))

        # Query-Passage similarities (Cosine, Euclidean)
        if vectorizer is not None:
            embdsQ = vectorizer.transform([df.at[i, "query"]])[0]
            embdsP = vectorizer.transform([passage])[0]

            df.at[i, "cosine"] = cosine(embdsQ, embdsP)
            df.at[i, "euclidean"] = np.linalg.norm(embdsQ-embdsP)

    print("Saving features")
    df.to_csv(f"features_{args.split}.txt", sep=" ", header=None, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create datasets for XGBoost.")

    parser.add_argument(
        '--split', type=str, required=True, choices=["train", "test"],
        help="Which dataset are you creating?")

    parser.add_argument(
        '--dataset', type=str, required=True,
        help="Path to dataset. (triples.train.small.tsv or BM25_ranking.txt).")

    parser.add_argument(
        '--indexes', type=str, required=True,
        help="Path to pyserini's indexes.")

    parser.add_argument(
        '--queries', type=str, required=True,
        help="Path to queries. [queries.train.tsv or queries.eval.tsv]")

    parser.add_argument(
        '--glove', type=str, default=None,
        help="Path to Glove's weights [e.g. glove.6B.300d.txt].")

    parser.add_argument(
        '--fasttext', type=str, default=None,
        help="Path to fastText's word2vec file.")

    args = parser.parse_args()

    if args.glove is not None and args.fasttext is not None:
        parser.error(
            "Please choose only one language model [--glove OR --fasttext]")

    main(args)
