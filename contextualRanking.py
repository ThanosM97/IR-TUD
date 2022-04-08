"""This module creates the rankings based on the distances of the embedding vectors."""
import argparse

import pandas as pd
import tensorflow as tf
from transformers import TFAutoModelForMaskedLM, AutoTokenizer

from pyserini.index import IndexReader
from tqdm import tqdm

# Get embeddings from the model for a specifix text
def get_embds(text, model, tokenizer):
    """Get the embeddings of a given text."""
    outputs = model(**tokenizer(text, return_tensors="tf"), output_hidden_states=True)
    last3 = outputs['hidden_states'][-3:]
    return tf.math.reduce_mean(last3, [0, 2])

def main(args):

    if args.model == "distilBert":
        model_checkpoint = "distilbert-base-uncased"
    elif args.model == "ALBERT":
        model_checkpoint = "albert-base-v2"

    # Download model
        print("Downloading pre-trained models..")
    model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    # Load finetuned model
    model = model.load_weights(args.model + "-full.h5")

    # Read data
    print("Reading datasets..")
    relevance = pd.read_csv(args.relevance, sep=" ", header=None, names=["qid", "Q0", "pid", "relevance"])

    ranking = pd.read_csv(args.ranking, sep="\t", header=None, names=["qid", "pid", "rank"])
    ranking = ranking[ranking["qid"].isin(relevance["qid"].unique())].drop("rank", axis=1)

    queries = pd.read_csv(args.queries, sep="\t", header=None, names=["qid", "query"])

    
    df = pd.merge(ranking, queries, how="left", on="qid")

    # Load pyserini's IndexReader
    index_reader = IndexReader(args.indexes)

    print("Calculating distance metrics..")
    cosine_similarity = tf.keras.losses.CosineSimilarity(
        axis=-1,
        reduction=tf.keras.losses.Reduction.SUM,
        name='cosine_similarity'
    )

    euclidean = lambda diff: tf.math.reduce_euclidean_norm(
        diff, axis=-1, name='euclidean_distance'
    )

    # Calculate cosine similarity and euclidean distance for all query-passage pairs
    for i in tqdm(range(df.shape[0])):
        passage = eval(
            index_reader.doc_raw(str(df.at[i, "pid"])))["contents"].lower()
        query = df.at[i, "query"]

        p_embds = get_embds(passage, model, tokenizer)
        q_embds = get_embds(query, model, tokenizer)

        df.at[i, "cosine"] = cosine_similarity(p_embds, q_embds).numpy()
        df.at[i, "euclidean"] = euclidean(p_embds - q_embds).numpy()[0]

    cosine = df.copy().drop(columns=["query", "euclidean"])
    cosine = cosine.groupby(["qid"]).apply(lambda x: x.sort_values(["cosine"], ascending = True)).reset_index(drop=True)
    cosine["rank"] = cosine.index % 1000 + 1

    euclidean = df.drop(columns=["query", "cosine"])
    euclidean = euclidean.groupby(["qid"]).apply(lambda x: x.sort_values(["euclidean"], ascending = True)).reset_index(drop=True)
    euclidean["rank"] = euclidean.index % 1000 + 1

    print("Saving distance metrics")
    cosine.drop(columns=["cosine"]).to_csv("cosine_ranking_"+ args.model +".txt", sep="\t", index=False, header=None)
    euclidean.drop(columns=["euclidean"]).to_csv("euclidean_ranking_"+ args.model +".txt", sep="\t", index=False, header=None)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Rerank passages given a fine-tuned contextual model.")

    parser.add_argument(
        '--relevance', type=str, required=True,
        help="Path to dataset with the query-passage relevance (e.g. 2019qrels-pass.txt).")

    parser.add_argument(
        '--indexes', type=str, required=True,
        help="Path to pyserini's indexes.")

    parser.add_argument(
        '--ranking', type=str, required=True,
        help="Path to initial ranking. (run.msmarco-passage.bm25-tuned+prf.topics.dl19-passage.txt)")

    parser.add_argument(
        '--queries', type=str, required=True,
        help="Path to eval queries. (queries.eval.tsv)")

    parser.add_argument(
        '--model', type=str, default="distilBert",
        help="Define the pre-trained model that will be used,'distilBert'(default) or 'ALBERT'")

    args = parser.parse_args()

    # if args.glove is not None and args.fasttext is not None:
    #     parser.error(
    #         "Please choose only one language model [--glove OR --fasttext]")

    main(args)