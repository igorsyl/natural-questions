import sys
import os
import re
import gzip
import glob
import json
import datetime
import pathlib
import math
import itertools
import pickle
import shutil
import logging
import random
import argparse
import functools
import multiprocessing
import time

from typing import Callable, Dict, List, Generator, Tuple
from dataclasses import dataclass
import collections

import tqdm
import torch
import tensorboardX
import transformers

import natural_questions.text_utils

logging.basicConfig(level=logging.INFO)
logging.getLogger("transformers").setLevel(logging.WARN)

logger = logging.getLogger(__name__)


class AttributeDict(dict):
    __getattr__ = dict.__getitem__
    __setattr__ = dict.__setitem__


data_path = pathlib.Path('data')
output_path = data_path / 'output'

# data_path = pathlib.Path('/home/ias/data/nq')
# output_path = data_path / 'work'

args = AttributeDict({
    'random_seed': 1029,
    # Maximum number of lines to read from input jsonl files. Used for testing.
    'max_jsonl_lines': None,
    # Maximum examples to generate. Used for testing.
    'max_examples': None,
    'output_path': output_path,
    'model_path': output_path / 'model',


    'train': False,
    # Estimated number of training examples. Used for learning rate scheduler and progress report.
    'train_size': 1_883_232,

    'train_jsonl_path': data_path / 'v1.0-simplified_simplified-nq-train.jsonl.gz',
    # 'train_jsonl_path': data_path / 'v1.0/train/*',

    'train_base_model': 'bert-base-uncased',
    # 'train_base_model': 'bert-large-uncased-whole-word-masking-finetuned-squad',
    # 'train_base_model': 'deepset/bert-large-uncased-whole-word-masking-squad2',
    # 'train_base_model': 'bert-large-uncased',


    'eval': False,
    # Estimated number of eval examples. Used for progress report only.
    'eval_size': 18_654,

    'eval_predictions_path': output_path / 'predictions.json',

    'eval_pred_jsonl_path': data_path / 'tiny-dev/nq-dev-sample.no-annot.jsonl.gz',
    # 'eval_pred_jsonl_path': data_path / 'v1.0/dev/nq-dev-??.jsonl.gz',

    'eval_gold_jsonl_path': data_path / 'tiny-dev/nq-dev-sample.jsonl.gz',
    # 'eval_gold_jsonl_path': data_path / 'v1.0/dev/nq-dev-??.jsonl.gz',


    # Maximum sequence level the model will support.
    'max_seq_len': 512,
    # Maximum token length for questions. Any questions longer than this will be truncated to this length.
    'max_question_len': 64,
    # When splitting up a long document into chunks, how much stride to take between chunks.
    'doc_stride': 128,

    # Number of processes for example generation.
    'num_procs': 10,
    # Number of training epochs.
    'num_epochs': 1,
    # Number of examples to chunk and randomize order.
    'chunk_size': 1_000,

    'train_devices': [torch.device(d) for d in [0]],
    'train_batch_size_per_gpu': 16,

    'eval_devices': [torch.device(d) for d in [0]],
    'eval_batch_size_per_gpu': 16,

    # Accumulate gradients for accumulation_steps before calling backward.
    'accumulation_steps': 2,
    # Record loss and learning rate to tensorboard every logging_steps.
    'logging_steps': 384,
    # Save model snapshot every save_steps.
    'save_steps': 500,

    # Optimizer and scheduler parameters
    'lr': 3e-5,
    'warmup': 0.1,

    'tokenizer': None,
})


@dataclass
class Example(object):
    example_id: int
    input_ids: List[int]
    # train fields
    start_pos: int
    end_pos: int
    # eval fields
    token_to_word_index: List[int]
    candidates: List[Dict]
    question_len: int
    doc_start: int


# Build map of HTML tags to tokens
# token_map = {'<P>': 'unused0', '<Table>': 'unused1', ...  '</H2>': 'unused26', '</H1>': 'unused27'}
html_tags = 'P Table Th Tr Td Ul Ol Dl Li Dd Dt H3 H2 H1'.split()
if sorted(set(html_tags)) != sorted(html_tags): raise Exception
html_tokens = [f'<{t}>' for t in html_tags] + [f'</{t}>' for t in html_tags]
token_map = dict(zip(html_tokens, (f'unused{n}' for n in range(100))))


def build_examples_from_jsonl_line(line: str, tokenizer=None, training=None) -> Example:
    if training is None:
        training = True

    nq_example = json.loads(line)
    if nq_example.get('document_tokens'):
        nq_example = natural_questions.text_utils.simplify_nq_example(nq_example)

    doc_words = nq_example['document_text'].split(' ')
    question_tokens = args.tokenizer.tokenize(nq_example['question_text'])[:args.max_question_len]

    # tokenized index of i-th original token corresponds to word_to_token_index[i]
    # if a token in original text is removed, its tokenized index indicates next token
    word_to_token_index = []
    token_to_word_index = []
    doc_tokens = []  # tokenized document text
    for i, word in enumerate(doc_words):
        word_to_token_index.append(len(doc_tokens))

        if word.startswith('<') and word.endswith('>'):
            word = token_map.get(word)
            if word is None:
                continue

        tokens = args.tokenizer.tokenize(word)
        for token in tokens:
            token_to_word_index.append(i)
            doc_tokens.append(token)

    annotation = None
    if training:
        annotation = nq_example['annotations'][0]
        long_answer = annotation['long_answer']

        if long_answer['candidate_index'] != -1:
            start_word = long_answer['start_token']
            end_word = long_answer['end_token']
        else:
            start_word = -1
            end_word = -1

        start_token = word_to_token_index[start_word]
        end_token = word_to_token_index[end_word]

    positive_examples = []
    negative_examples = []
    max_doc_len = args.max_seq_len - len(question_tokens) - 3  # [CLS], [SEP], [SEP]
    # take chunks with a stride of `doc_stride`
    for doc_start in range(0, len(doc_tokens), args.doc_stride):
        #doc_end = doc_start + min(max_doc_len, len(doc_tokens))
        chunk_doc_tokens = doc_tokens[doc_start : doc_start+max_doc_len]
        doc_end = doc_start + len(chunk_doc_tokens)
        input_tokens = ['[CLS]'] + question_tokens + ['[SEP]'] + chunk_doc_tokens + ['[SEP]']

        if training and doc_start <= start_token and end_token < doc_end:
            # if truncated document contains annotated range
            start_pos = start_token - doc_start + len(question_tokens) + 2
            end_pos = end_token - doc_start + len(question_tokens) + 2
        else:
            start_pos, end_pos = -1, -1

        example = Example(
            example_id=nq_example['example_id'],
            input_ids=args.tokenizer.convert_tokens_to_ids(input_tokens),
            # train fields
            start_pos=start_pos,
            end_pos=end_pos,
            # eval fields
            token_to_word_index=token_to_word_index,
            candidates=nq_example['long_answer_candidates'],
            question_len=len(question_tokens),
            doc_start=doc_start,
        )

        if example.end_pos > 0:
            positive_examples.append(example)
        else:
            negative_examples.append(example)

    if training:
        # For training, extract all positives examples and a uniform random sample of negative examples
        return positive_examples + random.sample(negative_examples, min(len(positive_examples), len(negative_examples)))
    else:
        # For evaluation all examples are negative (missing label)
        return negative_examples


def iter_jsonl_lines(file_glob):
    filenames = glob.glob(str(file_glob))
    for filename in filenames:
        with gzip.open(filename, 'rt') as f:
            yield from f


def iter_examples_from_jsonl(file_glob, training=None):
    lines = iter_jsonl_lines(file_glob)
    if args.max_jsonl_lines:
        lines = itertools.islice(lines, args.max_jsonl_lines)

    builder = functools.partial(build_examples_from_jsonl_line, training=training)

    if args.num_procs > 1:
        with multiprocessing.Pool(args.num_procs) as pool:
            example_set_iter = pool.imap_unordered(builder, lines)
            for example_set in example_set_iter:
                yield from example_set
    else:
        example_set_iter = map(builder, lines)
        for example_set in example_set_iter:
            yield from example_set


def collate_fn(examples):
    max_len = max([len(example.input_ids) for example in examples])
    input_ids = torch.zeros((len(examples), max_len), dtype=torch.int64)
    token_type_ids = torch.ones((len(examples), max_len), dtype=torch.int64)
    for i, example in enumerate(examples):
        input_ids[i, :len(example.input_ids)] = torch.tensor(example.input_ids)

        sep_index = example.input_ids.index(102) # 102 corresponds to [SEP]
        token_type_ids[i, :sep_index+1] = 0
        token_type_ids[i, sep_index+1:] = 1

    attention_mask = input_ids > 0

    start_positions = torch.tensor([example.start_pos for example in examples])
    end_positions = torch.tensor([example.end_pos for example in examples])

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "token_type_ids": token_type_ids,
        "start_positions": start_positions,
        "end_positions": end_positions,
    }, examples


def chunk_iterator(iterator, chunksize):
    chunk = []
    for item in iterator:
        if len(chunk) == chunksize:
            yield chunk
            chunk = []
        chunk.append(item)

    if chunk:
        yield chunk


class NqExamplesDataset(torch.utils.data.IterableDataset):
    def __init__(self, jsonl_path, training=None):
        examples = iter_examples_from_jsonl(jsonl_path, training=training)
        if args.max_examples:
            examples = itertools.islice(examples, args.max_examples)
        self.examples = examples

    def __iter__(self):
        return self.examples


def train_model():
    logging.info(f'using examples from {args.train_jsonl_path}')
    examples = iter_examples_from_jsonl(args.train_jsonl_path)

    if args.max_examples:
        examples = itertools.islice(examples, args.max_examples)

    examples_chunks = chunk_iterator(examples, args.chunk_size)

    args.model_path.mkdir(exist_ok=True)

    tb_path = args.output_path / 'tensorboard'
    if tb_path.exists():
        shutil.rmtree(tb_path)
    tb_writer = tensorboardX.SummaryWriter(logdir=str(tb_path))

    train_batch_size = args.train_batch_size_per_gpu * len(args.train_devices)

    bert_model = transformers.BertForQuestionAnswering.from_pretrained(args.train_base_model)
    model = bert_model

    param_optimizer = list(model.named_parameters())
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [
        {
            'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)],
            'weight_decay': 0.01
        },
        {
            'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)],
            'weight_decay': 0.0
        }
    ]

    num_train_optimization_steps = int(args.num_epochs * args.train_size / train_batch_size / args.accumulation_steps)
    num_warmup_steps = int(num_train_optimization_steps * args.warmup)

    optimizer = transformers.AdamW(optimizer_grouped_parameters, lr=args.lr, correct_bias=False)
    scheduler = transformers.get_linear_schedule_with_warmup(
        optimizer, num_warmup_steps=num_warmup_steps, num_training_steps=num_train_optimization_steps
    )

    model = model.to(args.train_devices[0])
    if len(args.train_devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.train_devices)

    model.zero_grad()
    model = model.train()

    start_time = time.time()
    step = 0
    tr_loss, logging_loss = 0.0, 0.0
    epochs = list(range(args.num_epochs))
    for epoch in tqdm.tqdm(epochs, desc='epoch'):
        with tqdm.tqdm(desc='examples', total=args.train_size, leave=True) as tqdm_examples:
            examples_chunks = tqdm.tqdm(examples_chunks, leave=False, desc='chunks', total=int(math.ceil(args.train_size / args.chunk_size)))
            for examples in examples_chunks:
                train_loader = torch.utils.data.DataLoader(
                    examples,
                    batch_size=train_batch_size,
                    shuffle=True,
                    collate_fn=collate_fn
                )
                train_loader = tqdm.tqdm(train_loader, desc='batches', leave=False, total=int(args.chunk_size/train_batch_size))
                for inputs, examples in train_loader:
                    inputs = dict((k,v.to(args.train_devices[0])) for (k,v) in inputs.items())
                    outputs = model(**inputs)
                    loss = outputs[0]
                    if len(args.train_devices) > 1:
                        loss = loss.mean()  # mean() to average on multi-gpu parallel training

                    loss.backward()
                    tr_loss += loss.item()

                    if step > 0 and step % args.accumulation_steps == 0:
                        optimizer.step()
                        scheduler.step()
                        model.zero_grad()

                    if args.save_steps and step % args.save_steps == 0:
                        bert_model.save_pretrained(str(args.model_path))

                    if step % args.logging_steps == 0:
                        tb_writer.add_scalar("lr", scheduler.get_last_lr()[0], step)
                        tb_writer.add_scalar("loss", (tr_loss - logging_loss) / args.logging_steps, step)
                        logging_loss = tr_loss

                    step += 1
                    tqdm_examples.update(train_batch_size)

    logging.info(f'save_pretrained {args.model_path}')
    bert_model.save_pretrained(str(args.model_path))


def eval_model():
    logging.info(f'using examples from {args.eval_pred_jsonl_path}')
    dataset = NqExamplesDataset(args.eval_pred_jsonl_path, training=False)

    eval_batch_size = args.eval_batch_size_per_gpu * len(args.eval_devices)

    eval_loader = torch.utils.data.DataLoader(
        dataset,
        batch_size=eval_batch_size,
        collate_fn=collate_fn
    )

    model = transformers.AutoModelForQuestionAnswering.from_pretrained(str(args.model_path))
    if len(args.eval_devices) > 1:
        model = torch.nn.DataParallel(model, device_ids=args.eval_devices)

    model = model.to(args.eval_devices[0])
    model.eval()

    predictions = {}
    with torch.no_grad() as _, tqdm.tqdm(desc='examples', total=args.eval_size) as pbar:
        # eval_loader = tqdm.tqdm(eval_loader, desc='batches', total=args.eval_size / eval_batch_size)
        for inputs, examples in eval_loader:
            inputs = dict((k,v.to(args.eval_devices[0])) for (k,v) in inputs.items())
            outputs = model(**inputs)
            start_logits, end_logits = outputs[1:3]

            for i, example in enumerate(examples):
                pbar.update(1)

                doc_slice = slice(example.question_len + 2, len(example.input_ids) - 1)
                start_logit, start_pos = torch.max(start_logits[i,doc_slice], dim=0)
                end_logit, end_pos = torch.max(end_logits[i,doc_slice], dim=0)

                start_token = start_pos.item() + example.doc_start if start_pos.item() >= 0 else -1
                end_token = end_pos.item() + example.doc_start if end_pos.item() >= 0 else -1

                cls_logit = start_logits[i, 0] + end_logits[i, 0]  # '[CLS]' logits
                score = (start_logit + end_logit - cls_logit).item()

                prediction = predictions.get(example.example_id, {})
                if not (score > prediction.get('long_answer_score', float('-inf'))):
                    continue

                long_start_word = -1
                long_end_word = -1

                if 0 <= start_token < end_token:
                    start_word = example.token_to_word_index[start_token]
                    end_word = example.token_to_word_index[end_token]

                    # Only consider spans that match any of the candidates.
                    # This should improve precision while keeping same recall.
                    for candidate in example.candidates:
                        if (candidate['start_token'] == start_word and end_word == candidate['end_token']):
                            long_start_word = candidate['start_token']
                            long_end_word = candidate['end_token']
                            break

                predictions[example.example_id] = {
                    'example_id': example.example_id,
                    'long_answer': {
                        'start_byte': -1, 'end_byte': -1,
                        'start_token': long_start_word, 'end_token': long_end_word
                    },
                    'long_answer_score': score,
                    'short_answers': [{
                        'start_byte': -1, 'end_byte': -1,
                        'start_token': -1, 'end_token': -1
                    }],
                    'short_answers_score': 0,
                    'yes_no_answer': 'NONE'
                }


    submission = { 'predictions': list(predictions.values()) }
    with open(args.eval_predictions_path, 'w') as f:
        json.dump(submission, f)

    import subprocess
    cmd = f'python -m natural_questions.nq_eval --predictions_path={args.eval_predictions_path} --gold_path={args.eval_gold_jsonl_path}'
    subprocess.check_call(cmd, shell=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true')
    parser.add_argument('--eval', action='store_true')
    parser_args = parser.parse_args(sys.argv[1:])

    args.update(vars(parser_args))
    for n in args.keys():
        if n.endswith('_path') and args[n]:
            args[n] = pathlib.Path(args[n])
    args.tokenizer = transformers.tokenization_auto.AutoTokenizer.from_pretrained(args.train_base_model)
    args.output_path.mkdir(parents=True, exist_ok=True)

    seed = args.random_seed
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True

    if args.train:
        train_model()

    if args.eval:
        eval_model()


if __name__ == '__main__':
    main()
