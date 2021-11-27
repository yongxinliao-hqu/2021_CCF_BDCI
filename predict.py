# Copyright (c) 2021 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from functools import partial
import argparse
import sys
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F
import paddlenlp as ppnlp
from paddlenlp.datasets import load_dataset
from paddlenlp.data import Stack, Tuple, Pad

from data import create_dataloader, read_text_pair, convert_example
from model import QuestionMatching

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--pretrain_model", type=str, required=True, help="The name of pretrain models")
parser.add_argument("--input_file", type=str, required=True, help="The full path of input file")
parser.add_argument("--result_file", type=str, required=True, help="The result file name")
parser.add_argument("--params_path", type=str, required=True, help="The path to model parameters to be loaded.")
parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument("--batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
args = parser.parse_args()
# yapf: enable


def predict(model, data_loader):
    """
    Predicts the data labels.

    Args:
        model (obj:`QuestionMatching`): A model to calculate whether the question pair is semantic similar or not.
        data_loaer (obj:`List(Example)`): The processed data ids of text pair: [query_input_ids, query_token_type_ids, title_input_ids, title_token_type_ids]
    Returns:
        results(obj:`List`): cosine similarity of text pairs.
    """
    batch_logits = []

    model.eval()

    with paddle.no_grad():
        for batch_data in data_loader:
            input_ids, token_type_ids = batch_data

            input_ids = paddle.to_tensor(input_ids)
            token_type_ids = paddle.to_tensor(token_type_ids)

            batch_logit, _ = model(
                input_ids=input_ids, token_type_ids=token_type_ids)

            batch_logits.append(batch_logit.numpy())

        batch_logits = np.concatenate(batch_logits, axis=0)

        return batch_logits


if __name__ == "__main__":
    paddle.set_device(args.device)

    if args.pretrain_model == "ernie-gram-zh": # 已测试
        pretrained_model = ppnlp.transformers.ErnieGramModel.from_pretrained(
            'ernie-gram-zh')
        tokenizer = ppnlp.transformers.ErnieGramTokenizer.from_pretrained(
            'ernie-gram-zh')
    elif args.pretrain_model == "ernie-1.0": # 已测试
        pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
            'ernie-1.0')
        tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(
            'ernie-1.0')
    elif args.pretrain_model == "ernie-2.0-en":# 已测试
        pretrained_model = ppnlp.transformers.ErnieModel.from_pretrained(
            'ernie-2.0-en')
        tokenizer = ppnlp.transformers.ErnieTokenizer.from_pretrained(
            'ernie-2.0-en')
    elif args.pretrain_model == "bert-base-chinese": # 已测试
        pretrained_model = ppnlp.transformers.BertModel.from_pretrained(
            'bert-base-chinese')
        tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained(
            'bert-base-chinese')
    elif args.pretrain_model == "bert-wwm-chinese": # 已测试
        pretrained_model = ppnlp.transformers.BertModel.from_pretrained(
            'bert-wwm-chinese')
        tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained(
            'bert-wwm-chinese')
    elif args.pretrain_model == "bert-wwm-ext-chinese":# 已测试
        pretrained_model = ppnlp.transformers.BertModel.from_pretrained(
            'bert-wwm-ext-chinese')
        tokenizer = ppnlp.transformers.BertTokenizer.from_pretrained(
            'bert-wwm-ext-chinese')
    elif args.pretrain_model == "chinese-xlnet-base":#不好使
        pretrained_model = ppnlp.transformers.XLNetModel.from_pretrained(
            'chinese-xlnet-base')
        tokenizer = ppnlp.transformers.XLNetTokenizer.from_pretrained(
            'chinese-xlnet-base')   
    elif args.pretrain_model == "chinese-xlnet-large": #不好使
        pretrained_model = ppnlp.transformers.XLNetModel.from_pretrained(
            'chinese-xlnet-large')
        tokenizer = ppnlp.transformers.XLNetTokenizer.from_pretrained(
            'chinese-xlnet-large')    
    elif args.pretrain_model == "chinese-electra-base": #不好使
        pretrained_model = ppnlp.transformers.ElectraModel.from_pretrained(
            'chinese-electra-base')
        tokenizer = ppnlp.transformers.ElectraTokenizer.from_pretrained(
            'chinese-electra-base')    
    elif args.pretrain_model == "roberta-wwm-ext-large": #GPU 0.5的跑不动，需要用batch size 16
        pretrained_model = ppnlp.transformers.RobertaModel.from_pretrained(
            'roberta-wwm-ext-large')
        tokenizer = ppnlp.transformers.RobertaTokenizer.from_pretrained(
            'roberta-wwm-ext-large')    
    elif args.pretrain_model == "roberta-wwm-ext":# 用batch size 16
        pretrained_model = ppnlp.transformers.RobertaModel.from_pretrained(
            'roberta-wwm-ext')
        tokenizer = ppnlp.transformers.RobertaTokenizer.from_pretrained(
            'roberta-wwm-ext')    
    else:
        print("pretrain model erros")

    trans_func = partial(
        convert_example,
        tokenizer=tokenizer,
        max_seq_length=args.max_seq_length,
        is_test=True)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # input_ids
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # segment_ids
    ): [data for data in fn(samples)]

    test_ds = load_dataset(
        read_text_pair, data_path=args.input_file, is_test=True, lazy=False)

    test_data_loader = create_dataloader(
        test_ds,
        mode='predict',
        batch_size=args.batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    model = QuestionMatching(pretrained_model)

    if args.params_path and os.path.isfile(args.params_path):
        state_dict = paddle.load(args.params_path)
        model.set_dict(state_dict)
        print("Loaded parameters from %s" % args.params_path)
    else:
        raise ValueError(
            "Please set --params_path with correct pretrained model file")

    y_probs = predict(model, test_data_loader)
    y_preds = np.argmax(y_probs, axis=1)
    
    with open(args.result_file, 'w', encoding="utf-8") as f:
        for y_pred in y_preds:
            f.write(str(y_pred) + "\n")