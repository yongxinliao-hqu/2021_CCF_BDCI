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
import os
import random
import time

import numpy as np
import paddle
import paddle.nn.functional as F

import paddlenlp as ppnlp
from paddlenlp.data import Stack, Tuple, Pad
from paddlenlp.datasets import load_dataset
from paddlenlp.transformers import LinearDecayWithWarmup

from data import create_dataloader, read_text_pair, convert_example
from model import QuestionMatching

import matplotlib.pyplot as plt

# yapf: disable
parser = argparse.ArgumentParser()
parser.add_argument("--early_stop", default= "None", type=str, required=False, help="If the model cannot provide better results after n times of evaluation, stop!")
parser.add_argument("--pretrain_model", type=str, required=True, help="The name of pretrain models")
parser.add_argument("--train_set", type=str, required=True, help="The full path of train_set_file")
parser.add_argument("--dev_set", type=str, required=True, help="The full path of dev_set_file")
parser.add_argument("--save_dir", default='./checkpoint', type=str, help="The output directory where the model checkpoints will be written.")
parser.add_argument("--max_seq_length", default=256, type=int, help="The maximum total input sequence length after tokenization. "
    "Sequences longer than this will be truncated, sequences shorter will be padded.")
parser.add_argument('--max_steps', default=-1, type=int, help="If > 0, set total number of training steps to perform.")
parser.add_argument("--train_batch_size", default=32, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--eval_batch_size", default=128, type=int, help="Batch size per GPU/CPU for training.")
parser.add_argument("--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam.")
parser.add_argument("--weight_decay", default=0.0, type=float, help="Weight decay if we apply some.")
parser.add_argument("--epochs", default=3, type=int, help="Total number of training epochs to perform.")
parser.add_argument("--eval_step", default=100, type=int, help="Step interval for evaluation.")
parser.add_argument('--save_step', default=10000, type=int, help="Step interval for saving checkpoint.")
parser.add_argument("--warmup_proportion", default=0.0, type=float, help="Linear warmup proption over the training process.")
parser.add_argument("--init_from_ckpt", type=str, default=None, help="The path of checkpoint to be loaded.")
parser.add_argument("--seed", type=int, default=1000, help="Random seed for initialization.")
parser.add_argument('--device', choices=['cpu', 'gpu'], default="gpu", help="Select which device to train model, defaults to gpu.")
parser.add_argument("--rdrop_coef", default=0.0, type=float, help="The coefficient of" 
    "KL-Divergence loss in R-Drop paper, for more detail please refer to https://arxiv.org/abs/2106.14448), if rdrop_coef > 0 then R-Drop works")

args = parser.parse_args()
# yapf: enable

# 绘出指定数据的变化曲线（一条）
def draw_line(y, steps):
    x = []
    for i in range (1,len(y)+1):
        x.append(i*steps)
    y = y
    
    plt.plot(x,y)

    #显示图形
    plt.show()

# 绘出指定数据的变化曲线（两条）
def draw_lines(lable_1,label_2, y1, y2, filename):
    x1 = []
    for i in range (1,len(y1)+1):
        x1.append(i * 10)
    x2 = []
    for i in range (1,len(y2)+1):
        x2.append(i * args.eval_step)

    l1, = plt.plot(x1,y1,color='red',linewidth=1)
    l2, = plt.plot(x2,y2,color='blue',linewidth=1)
    
    plt.legend(handles=[l1,l2],labels=[lable_1,label_2],loc='best')

    plt.xlabel('x')
    plt.ylabel('y')

    plt.savefig(filename)

    #显示图形
    plt.show()

# 计算总共的训练steps
def caculate_steps(data_path):
    with open(data_path, 'r', encoding='utf-8') as f:
        count = 0
        for line in f:
            if line!="\n":
                count = count + 1
    return count

def print_total_steps(data_path, count, total_steps, epochs, train_batch_size):
    print("\n" + "="*30)
    print("The number of lines in file " + data_path + " is " + str(count))
    print("epochs: " + str(epochs))
    print("train_batch_size: " + str(train_batch_size))
    print("The total number of steps is " + str(int(total_steps)))
    print("="*30 + "\n")


def set_seed(seed):
    """sets random seed"""
    random.seed(seed)
    np.random.seed(seed)
    paddle.seed(seed)


evaluation_results = ["evaluation results are as follows:"]
train_losses = []
eval_losses = []
train_accuracies = []
eval_accuracies = []

@paddle.no_grad()
def evaluate(global_step, loss, model, criterion, metric, data_loader, evaluation_results, train_acc):
    """
    Given a dataset, it evals model and computes the metric.

    Args:
        model(obj:`paddle.nn.Layer`): A model to classify texts.
        data_loader(obj:`paddle.io.DataLoader`): The dataset loader which generates batches.
        criterion(obj:`paddle.nn.Layer`): It can compute the loss.
        metric(obj:`paddle.metric.Metric`): The evaluation metric.
    """
    model.eval()
    metric.reset()
    losses = []
    total_num = 0

    for batch in data_loader:
        input_ids, token_type_ids, labels = batch
        total_num += len(labels)
        logits, _ = model(input_ids=input_ids, token_type_ids=token_type_ids, do_evaluate=True)
        loss = criterion(logits, labels)
        losses.append(loss.numpy())
        correct = metric.compute(logits, labels)
        metric.update(correct)
        accu = metric.accumulate()
    
    for each_result in evaluation_results:
        print(each_result)
    
    evaluation_result = "== steps: %d, train_loss: %.4f, dev_loss: %.4f, dev acc: %.4f, train acc: %.4f, num: %d ==" % (global_step, loss, np.mean(losses), accu, train_acc, total_num)
    print(evaluation_result)

    #print("==== steps:{}, train_loss: {:.5}, dev_loss: {:.5}, acc: {:.5}, num:{} ====".format(global_step, loss, np.mean(losses), accu, total_num))
    model.train()
    metric.reset()
    return accu, evaluation_result, np.mean(losses)


def do_train():
    early_stop_count = 0
    paddle.set_device(args.device)
    rank = paddle.distributed.get_rank()
    if paddle.distributed.get_world_size() > 1:
        paddle.distributed.init_parallel_env()

    set_seed(args.seed)

    train_ds = load_dataset(
        read_text_pair, data_path=args.train_set, is_test=False, lazy=False)

    dev_ds = load_dataset(
        read_text_pair, data_path=args.dev_set, is_test=False, lazy=False)

    count_line = caculate_steps(args.train_set)
    total_steps = int(count_line)*int(args.epochs)/int(args.train_batch_size)
    print_total_steps(args.train_set, count_line, total_steps, args.epochs, args.train_batch_size)

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
        max_seq_length=args.max_seq_length)

    batchify_fn = lambda samples, fn=Tuple(
        Pad(axis=0, pad_val=tokenizer.pad_token_id),  # text_pair_input
        Pad(axis=0, pad_val=tokenizer.pad_token_type_id),  # text_pair_segment
        Stack(dtype="int64")  # label
    ): [data for data in fn(samples)]

    train_data_loader = create_dataloader(
        train_ds,
        mode='train',
        batch_size=args.train_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)

    dev_data_loader = create_dataloader(
        dev_ds,
        mode='dev',
        batch_size=args.eval_batch_size,
        batchify_fn=batchify_fn,
        trans_fn=trans_func)


    model = QuestionMatching(pretrained_model, rdrop_coef=args.rdrop_coef)

    if args.init_from_ckpt and os.path.isfile(args.init_from_ckpt):
        state_dict = paddle.load(args.init_from_ckpt)
        model.set_dict(state_dict)

    model = paddle.DataParallel(model)

    num_training_steps = len(train_data_loader) * args.epochs

    lr_scheduler = LinearDecayWithWarmup(args.learning_rate, num_training_steps,
                                         args.warmup_proportion)

    # Generate parameter names needed to perform weight decay.
    # All bias and LayerNorm parameters are excluded.
    decay_params = [
        p.name for n, p in model.named_parameters()
        if not any(nd in n for nd in ["bias", "norm"])
    ]
    optimizer = paddle.optimizer.AdamW(
        learning_rate=lr_scheduler,
        parameters=model.parameters(),
        weight_decay=args.weight_decay,
        apply_decay_param_fun=lambda x: x in decay_params)

    criterion = paddle.nn.loss.CrossEntropyLoss()
    
    metric = paddle.metric.Accuracy()

    global_step = 0
    best_accuracy = 0.0

    tic_train = time.time()
    to_break = False
    for epoch in range(1, args.epochs + 1): 
        for step, batch in enumerate(train_data_loader, start=1):
            input_ids, token_type_ids, labels = batch
            logits1, kl_loss = model(input_ids=input_ids, token_type_ids=token_type_ids)
            correct = metric.compute(logits1, labels)
            metric.update(correct)
            acc = metric.accumulate()

            ce_loss = criterion(logits1, labels)
            if kl_loss > 0:
                # 原始loss
                # loss = ce_loss + kl_loss * args.rdrop_coef 
                # 修改后的loss
                loss = ce_loss*(1-args.rdrop_coef) + kl_loss * args.rdrop_coef 
            else:
                #原始loss
                loss = ce_loss 
                # loss = ce_loss*ce_loss # 这里做了修改
                
            
            global_step += 1
            if global_step % 10 == 0 and rank == 0:
                train_losses.append(float("%.4f" %(loss)))
                train_accuracies.append(float("%.4f" %(acc)))

                if args.rdrop_coef != 0.0:
                    print("global step %d, epoch: %d, batch: %d, loss: %.4f, ce_loss: %.4f., kl_loss: %.4f, accu: %.4f, speed: %.2f step/s" % (global_step, epoch, step, loss, ce_loss, kl_loss, acc, 10 / (time.time() - tic_train)))
                else:
                    print("global step %d, epoch: %d, batch: %d, loss: %.4f, accu: %.4f, speed: %.2f step/s" % (global_step, epoch, step, loss, acc, 10 / (time.time() - tic_train)))
                
                tic_train = time.time()

            loss.backward()
            optimizer.step()
            lr_scheduler.step()
            optimizer.clear_grad()

            if global_step % args.eval_step == 0 and rank == 0:

                accuracy, evaluation_result, eval_loss= evaluate(global_step, loss, model, criterion, metric, dev_data_loader, evaluation_results, acc)
                
                if accuracy > best_accuracy:
                # if True:
                    early_stop_count = 0
                    save_dir = os.path.join(args.save_dir, "model_%d" % global_step)
                    if not os.path.exists(save_dir):
                        os.makedirs(save_dir)
                    save_param_path = os.path.join(save_dir, 'model_state.pdparams')
                    paddle.save(model.state_dict(), save_param_path)
                    tokenizer.save_pretrained(save_dir)
                    best_accuracy = accuracy

                    evaluation_results.append(evaluation_result)
                else:
                    early_stop_count = early_stop_count + 1
                    print("="*2 + " early_stop_count: " + str(early_stop_count) + " max("+ args.early_stop + ") "+"="*2)

                eval_losses.append(float("%.4f" %(eval_loss)))
                eval_accuracies.append(float("%.4f" %(accuracy)))
            
            if global_step ==  args.max_steps:
                print("==================== train finished! Because of max steps arrived ====================")
                print("Trained steps: " + str(global_step) + " max steps: " + str(args.max_steps))
                to_break = True
                break
            elif args.early_stop != "None" and early_stop_count >= int(args.early_stop):
                print("==================== train finished! Because of early stop! ====================")
                print("Trained steps: " + str(global_step) + " Total steps: " + str(int(total_steps)))
                to_break = True
                break
        if to_break == True:
            break
    print("==================== train finished!  ====================")
    print("Trained steps: " + str(global_step) + " Total steps: " + str(int(total_steps)))
    print("======================================================================================")
    for each_result in evaluation_results:
        print(each_result)

    print("="*15 + " train_losses "+ "="*15 )
    print(train_losses)
    print("="*15 + " eval_losses "+ "="*15 )
    print(eval_losses)
    print("="*15 + " train_accuracies "+ "="*15 )
    print(train_accuracies)
    print("="*15 + " eval_accuracies "+ "="*15 )
    print(eval_accuracies)

    draw_lines("train loss","eval loss", train_losses, eval_losses, "work/loss.jpg")
    draw_lines("train acc","eval acc", train_accuracies, eval_accuracies, "work/acc.jpg")


if __name__ == "__main__":
    do_train()