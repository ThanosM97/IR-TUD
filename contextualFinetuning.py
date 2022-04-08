"""This module finetunes the pre-trained model to the corpus of our choice."""
import argparse
import collections
import math
import functools

import numpy as np
import tensorflow as tf

from transformers import TFAutoModelForMaskedLM, AutoTokenizer
from transformers import DataCollatorForLanguageModeling, create_optimizer
from transformers.data import default_data_collator
from datasets import load_dataset


def tokenize_function(examples, tokenizer):
    """Tokenize the input dataset passages."""
    result = tokenizer(examples["text"])
    if tokenizer.is_fast:
        result["word_ids"] = [result.word_ids(i) for i in range(len(result["input_ids"]))]
    return result

def group_texts(examples):
    """Group all texts of the corpus and break into chunks."""
    chunk_size = 128
    # Concatenate all texts
    concatenated_examples = {k: sum(examples[k], []) for k in examples.keys()}
    # Compute length of concatenated texts
    total_length = len(concatenated_examples[list(examples.keys())[0]])
    # We drop the last chunk if it's smaller than chunk_size
    total_length = (total_length // chunk_size) * chunk_size
    # Split by chunks of max_len
    result = {
        k: [t[i : i + chunk_size] for i in range(0, total_length, chunk_size)]
        for k, t in concatenated_examples.items()
    }
    # Create a new labels column
    result["labels"] = result["input_ids"].copy()
    return result


def whole_word_masking_data_collator(features, tokenizer):
    """Data collator to mask words together."""
    wwm_probability = 0.2

    for feature in features:
        word_ids = feature.pop("word_ids")

        # Create a map between words and corresponding token indices
        mapping = collections.defaultdict(list)
        current_word_index = -1
        current_word = None
        for idx, word_id in enumerate(word_ids):
            if word_id is not None:
                if word_id != current_word:
                    current_word = word_id
                    current_word_index += 1
                mapping[current_word_index].append(idx)

        # Randomly mask words
        mask = np.random.binomial(1, wwm_probability, (len(mapping),))
        input_ids = feature["input_ids"]
        labels = feature["labels"]
        new_labels = [-100] * len(labels)
        for word_id in np.where(mask)[0]:
            word_id = word_id.item()
            for idx in mapping[word_id]:
                new_labels[idx] = labels[idx]
                input_ids[idx] = tokenizer.mask_token_id

    return default_data_collator(features)

def main(args):
    if args.model == "distilBert":
        model_checkpoint = "distilbert-base-uncased"
    elif args.model == "ALBERT":
        model_checkpoint = "albert-base-v2"

    # Download model
    print("Downloading pre-trained models..")
    model = TFAutoModelForMaskedLM.from_pretrained(model_checkpoint)
    tokenizer = AutoTokenizer.from_pretrained(model_checkpoint)

    print("Loading the datasets..")
    dataset = load_dataset('text', 
                        data_files=[args.dataset, 
                                    args.queries])

    tokenize_partial = functools.partial(tokenize_function, tokenizer=tokenizer)
    
    print("Tokenizing the dataset..")
    # Tokenize corpus
    tokenized_datasets = dataset.map(
        tokenize_partial, batched=True, remove_columns=["text"])


    # Group all queries and passages and break into chunks
    lm_datasets = tokenized_datasets.map(group_texts, batched=True)

    # Finetune the model to our corpus

    # Mask random tokens
    data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm_probability=0.15)

    samples = [lm_datasets["train"][i] for i in range(2)]
    for sample in samples:
        _ = sample.pop("word_ids")

    for chunk in data_collator(samples)["input_ids"]:
        print(f"\n'>>> {tokenizer.decode(chunk)}'")

    # Mask whole words together
    samples = [lm_datasets["train"][i] for i in range(2)]
    batch = whole_word_masking_data_collator(samples, tokenizer)

    for chunk in batch["input_ids"]:
        print(f"\n'>>> {tokenizer.decode(chunk)}'")

    # Downsample the size of training set
    train_size = 10_000
    test_size = int(0.1 * train_size)

    downsampled_dataset = lm_datasets["train"].train_test_split(
        train_size=train_size, test_size=test_size, seed=42
    )
    downsampled_dataset

    # Split dataset into train and eval sets
    tf_train_dataset = downsampled_dataset["train"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=data_collator,
        shuffle=True,
        batch_size=32,
    )

    tf_eval_dataset = downsampled_dataset["test"].to_tf_dataset(
        columns=["input_ids", "attention_mask", "labels"],
        collate_fn=data_collator,
        shuffle=False,
        batch_size=32,
    )

    # Optimize model
    num_train_steps = len(tf_train_dataset)
    optimizer, schedule = create_optimizer(
        init_lr=2e-5,
        num_warmup_steps=1_000,
        num_train_steps=num_train_steps,
        weight_decay_rate=0.01,
    )
    model.compile(optimizer=optimizer)

    # Train in mixed-precision float16
    tf.keras.mixed_precision.set_global_policy("mixed_float16")

    model.fit(tf_train_dataset, validation_data=tf_eval_dataset)
    
    eval_loss = model.evaluate(tf_eval_dataset)
    print(f"Perplexity: {math.exp(eval_loss):.2f}")


    # Save Model to extract Embeddings
    print("Saving fine-tuned model")
    model.save_weights(args.model + "-full.h5", save_format="h5")





if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Find Distances from contextual embeddings.")

    parser.add_argument(
        '--dataset', type=str, required=True,
        help="Path to dataset (e.g. triples.train.small.tsv).")

    parser.add_argument(
        '--queries', type=str, required=True,
        help="Path to training queries. (queries.train.tsv)")

    parser.add_argument(
        '--model', type=str, default="distilBert",
        help="Define the pre-trained model that will be used 'distilBert' or 'ALBERT'")

    args = parser.parse_args()

    main(args)