from datasets import load_dataset, load_metric
from transformers import BertForSequenceClassification,BertTokenizer, Trainer, TrainingArguments,SchedulerType, EvalPrediction, DataCollator,AutoTokenizer
from torch.utils.data import DataLoader
import torch
import torch.nn as nn
from functools import partial
import numpy as np
import argparse


def collate_fn(examples, tokenizer):
  # print(examples)
  # samples  = {}
  labels = []
  ls_question_1 = []
  ls_question_2 = []
  for example in examples:
    ls_question_1.append(example['question1'])
    ls_question_2.append(example['question2'])
    labels.append(example['label'])

  tok_out = tokenizer(ls_question_1, ls_question_2, padding = 'longest', return_tensors = 'pt')
  tok_out['labels'] = torch.tensor(labels)

  return tok_out

acc = lambda metric, eval_out: metric.compute(predictions = np.argmax(eval_out.predictions,axis = -1), references= eval_out.label_ids)

class Dataset:
  def __init__(self,ds):
    self.ds = ds

  def __len__(self): return len(self.ds)

  def __getitem__(self, i): return self.ds[i]


def main():

    parser = argparse.ArgumentParser()

    parser.add_argument('--output_dir', help = 'The path to directory where you want to save you finetuned model', type = str, default = '../fine_tuned_model')
    parser.add_argument('--dataset', help = 'Dataset from GLUE benchmark you want to train on', type = str, default= 'qqp')
    parser.add_argument('--train_epochs', help = 'No of epochs to train for', type = int, default = 1)
    parser.add_argument('--gradient_accumulation_steps', help = 'No of batches to pass before the gradient update', type = int, default = 8)
    parser.add_argument('--train_bs', help = 'batch size for training', type = int, default = 32)
    parser.add_argument('--eval_bs', help = 'batch size for evaluation', type = int, default = 128)
    parser.add_argument('--learning_rate', help = 'learning rate for training', type = float, default = 5e-05)
    parser.add_argument('--tokenizer_type', help = 'whether to use cased or uncased bert tokenizer based on your dataset', type = str, default = 'cased')

    args = parser.parse_args()

    metric = load_metric('accuracy')
    dataset = load_dataset('glue', args.dataset)
    tokenizer = BertTokenizer.from_pretrained('bert-base-cased') if args.tokenizer_type == 'cased' else BertTokenizer.from_pretrained('bert-base-uncased')

    model = BertForSequenceClassification.from_pretrained('bert-base-cased', num_labels = dataset['train'].features['label'].num_classes)

    arguments = TrainingArguments(output_dir= args.output_dir,
                                  evaluation_strategy= 'steps',
                                  eval_accumulation_steps= 1,
                                  learning_rate = args.learning_rate,
                                  num_train_epochs= args.train_epochs,
                                  save_steps = 100,
                                  gradient_accumulation_steps = args.gradient_accumulation_steps,
                                  lr_scheduler_type = 'cosine',
                                  warmup_ratio = 0.2,
                                  logging_strategy = 'steps',    
                                  logging_steps = 100,
                                  save_total_limit = 1,
                                  fp16 = True,
                                  metric_for_best_model = 'accuracy',
                                  greater_is_better = True,
                                  per_device_train_batch_size = args.train_bs,
                                  per_device_eval_batch_size = args.eval_bs)


    trainer = Trainer(model,
                      args = arguments,
                      compute_metrics = partial(acc, metric),
                      train_dataset = Dataset(dataset['train']),
                      eval_dataset = Dataset(dataset['validation']),
                      data_collator = partial(collate_fn, tokenizer = tokenizer)
                      )

    trainer.train()


if __name__ == '__main__':
  main()

