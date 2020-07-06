# Transformers Model for Google's Natural Questions Dataset

## Overview

I trained BertForQuestionAnswering using bert-base-uncased with maximum sequence length of 512.

I chose bert-base to develop quickly and would use bert-large-uncased or bert-large-uncased-whole-word-masking-finetuned-squad given more training capacity to achieve better performance.

I chose the maximum sequence length supported in order to fit as many long answers as possible. There are a few lines with long answers which are too long to fit in the encoded input sequence and these examples would always be misclassified.

Each line of the jsonl dataset was converted into several training examples. Each document was split into overlapping chunks and converted into positive examples which fully contain the long answer and negative examples, otherwise.

An alternative approach to generating examples is to generate one example for each long answer candidate. Since any predicted answer must be a candidate, this approach exploits the information available in the set of candidates. Given more time I would explore this alternative example strategy.

Since the dataset is negatively un-balanced, I sampled all positive examples and up to the same number of negative examples to try to achieve a 1:1 positive-to-negative ratio.

Since BERT is trained on plain text the vocabulary does not contain HTML tokens. So, I tokenized the most common HTML tags using consecutive unused tokens to aid the model in understanding the document structure.

I used the formula for score as outlined in the BERT-Joint paper since I suspect the authors experimented with several other output combinations.

I believe that negative sampling is critical to achiving better performance. I don't expect random samping of negative examples to be optimal. Instead, I expect a better negative sampling strategy is to weight negative samples by a difficulty score. In this case, difficult negative examples would be more likely to be sampled. A strategy is to train the model using uniform random negative sampling and use the score output to train the same model using weighted random negative sampling. This process could be repeated a few times to converge to a stable assignment of negative weights.

## Honest Opinion

Given I started with only superficial knowledge of the transformers library, this project gave me a great opportunity to learn about implementation details.  Even though there are many published solutions such as the baseline BERT-Joint project and solutions from a Kaggle competition, I believe it's non-trivial to put together a working solution.

## Favorite Charity

TIL about code.org =D

## Installation

Download and install Miniconda

```
curl -O https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh
bash Miniconda3-latest-Linux-x86_64.sh
```

Create a python environment and install packages

```
conda create -y -n py36 python=3.6
conda activate py36
conda install -y pytorch transformers tensorboardX tqdm
```

Clone the project repo

```
git clone git@github.com:igorsyl/natural-questions.git
cd natural-questions
```

## Evaluation Results

Here are the evaluation results for tiny-dev dataset using bert-base-uncased:

```
  "long-best-threshold-f1": 0.5063291139240507,
  "long-best-threshold-precision": 0.7272727272727273,
  "long-best-threshold-recall": 0.3883495145631068,
  "long-best-threshold": 4.020364761352539,
  "long-recall-at-precision>=0.5": 0.3883495145631068,
  "long-precision-at-precision>=0.5": 0.7272727272727273,
  "long-recall-at-precision>=0.75": 0.34951456310679613,
  "long-precision-at-precision>=0.75": 0.75,
  "long-recall-at-precision>=0.9": 0.22330097087378642,
  "long-precision-at-precision>=0.9": 0.92,
```

For comparison, here are the results for the baseline BER-Join on the full dev dataset using bert-large-uncased:
```
  "long-best-threshold-f1": 0.6168,
  "short-best-threshold-f1": 0.5620,
```

## Replicating Results

Create a directory to hold dataset and training model:

```
mkdir data
cd data
```

Download the simplified version of the natural questions dataset:

```
curl -O https://storage.cloud.google.com/natural_questions/v1.0-simplified/simplified-nq-train.jsonl.gz
```

Download the tiny dev dataset:

```
gsutil cp -R gs://bert-nq/tiny-dev .
```

Optionally, download the full version of the dataset:
```
gsutil -m cp -r gs://natural_questions data
```

### Training

```
python run.py --train
```

### Evaluating

This will generate predictions and run the nq_eval script

```
python run.py --eval
```
