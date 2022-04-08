# Learning To Rank with deep features and different embedding representations for the passage re-ranking task of the TREC 2019 Deep Learning track.
This repository hosts the code for the Applied NLP project of the Information Retrieval (IN4325) course of TU Delft.

## Pipeline
Start by creating training and test dataset by running script:

```
usage: create_dataset.py [-h] --split {train,test} --dataset DATASET --indexes INDEXES --queries QUERIES
                         [--glove GLOVE] [--fasttext FASTTEXT]

Create datasets for XGBoost.

optional arguments:
  -h, --help            show this help message and exit
  --split {train,test}  Which dataset are you creating?
  --dataset DATASET     Path to dataset. (triples.train.small.tsv or BM25_ranking.txt).
  --indexes INDEXES     Path to pyserini's indexes.
  --queries QUERIES     Path to queries. [queries.train.tsv or queries.eval.tsv]
  --glove GLOVE         Path to Glove's weights [e.g. glove.6B.300d.txt].
  --fasttext FASTTEXT   Path to fastText's word2vec file. 

```

  
Make 90-10 query aware split into training and development sets:

```  
usage: train_dev_split.py [-h] -d D

Split dataset to train/dev sets.

optional arguments:
  -h, --help  show this help message and exit
  -d D        Path to dataset.
  
```

Train an XGBoost classifier by specifying the --train TRAIN path and re-rank the passages by specifying the --test TEST path:
 
```
usage: xgb_ranking.py [-h] [--train TRAIN] [--val VAL] [--save SAVE] [--test TEST] [--checkpoint CHECKPOINT]

Train and test XGBoost classifier.

optional arguments:
  -h, --help            show this help message and exit
  --train TRAIN         Path to train dataset.
  --val VAL             Path to validation dataset.
  --save SAVE           Filename for the saved model.
  --test TEST           Path to test dataset.
  --checkpoint CHECKPOINT
                        Path to checkpoint model.
                        
```

Fine-tune the Huggingface's DistilBert and ALBERT models on the MS MARCO dataset:

```
usage: contextualFinetuning.py [-h] --dataset DATASET --queries QUERIES [--model MODEL]

Find Distances from contextual embeddings.

optional arguments:
  -h, --help         show this help message and exit
  --dataset DATASET  Path to dataset (e.g. triples.train.small.tsv).
  --queries QUERIES  Path to training queries. (queries.train.tsv)
  --model MODEL      Define the pre-trained model that will be used 'distilBert' or 'ALBERT'
 
```

Rerank original rankings by using distance metrics between contextual embeddings:

```
usage: contextualRanking.py [-h] --relevance RELEVANCE --indexes INDEXES --ranking RANKING --queries QUERIES
                            [--model MODEL]

Rerank passages given a fine-tuned contextual model.

optional arguments:
  -h, --help            show this help message and exit
  --relevance RELEVANCE
                        Path to dataset with the query-passage relevance (e.g. 2019qrels-pass.txt).
  --indexes INDEXES     Path to pyserini's indexes.
  --ranking RANKING     Path to initial ranking. (run.msmarco-passage.bm25-tuned+prf.topics.dl19-passage.txt)
  --queries QUERIES     Path to eval queries. (queries.eval.tsv)
  --model MODEL         Define the pre-trained model that will be used,'distilBert'(default) or 'ALBERT'
  
```
